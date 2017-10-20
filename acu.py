import logUtils
import main.analysis.singleMetric as singleMetric
from constants import app, event
from decision.actions.actionOption import ActionOption
#from decision.marge.src.forecast import iterateThroughForecast
from decision.models import PlannedAction, ConstantInstance, Forecast, PlannedActionMembership, PlannedActionLog
from main.models import MetricDefinition, UserProfile

import logging
import math  # Can be used in eval below. Do not remove.
import numpy as np
import random
import time
import traceback
from datetime import datetime, timedelta  # Can be used in eval.
from django.db.models import Q
import pickle

logger = logUtils.Logger(logging.getLogger("application"), app.DECISION_ENGINE)

def getMembershipForUserProfile(userProfile):
  # Get memberships via userProfile.
  ret = PlannedActionMembership.objects.filter(userProfile=userProfile)
  ret = list(ret)

  # Get memberships via group.
  groupMemberships = PlannedActionMembership.objects.filter(group__isnull=False)
  for gp in groupMemberships:
    if userProfile.user.groups.filter(name=gp.group.name).exists():
      ret.append(gp)

  return ret

def getMembershipForUserProfileAndAction(plannedAction, userProfile):
  # Multiple corner cases.
  # 1. If membership via userProfile and group - Use userProfile membership.
  # 2. If membership via multiple groups - Use the first group membership.
  ret = PlannedActionMembership.objects.filter(action=plannedAction, userProfile=userProfile)

  if not ret:
    # Search for membership via group.
    for gp in PlannedActionMembership.objects.filter(action=plannedAction, group__isnull=False):
      if userProfile.user.groups.filter(name=gp.group.name).exists():
        ret = gp
        break
  else:
    ret = ret[0]

  # To handle the case of userProfile vs group opted out.
  if ret and ret.optedOut == True:
    ret = None

  return ret

# TODO: What is the difference between getAvailbleOptions and
# getRecommendedOptions other than post job option filtering?
# Can we remove getAvailableOptions?
def getApiResults(user, hospital, workArea, execDt, recDt=None):
  userProfile = UserProfile.objects.get(user=user)
  ''' 
  with open("getApiResults.txt", "a") as text_file:
    text_file.write(str(recDt)+str(user)+str(userProfile)+str(workArea)+str(execDt) + "\n")
  '''
  if recDt == None:
    recDt = datetime.now()

  # Make sure user has access to the hospital.
  userHospitalPks = [x['pk'] for x in userProfile.hospitals.values('pk').all()]
  if hospital.pk in userHospitalPks:
    ret = {
      'hospital': hospital.pk,
      'workArea': workArea,
      'recDt': recDt,
      'execDt': execDt,
      'options': [x for x in getAvailableOptions(userProfile, hospital, workArea, execDt, recDt)]
    }
  else:
    ret = {'error': 1, 'message': 'Bad permissions for this hospital.'}

  return ret


# pushing = we are pushing recommendations, not getting results for a query.
def getRecommendedOptions(userProfile, hospital, execDt, recDt=None, pushing=False, action2cachedOptions=None):
  action2options = {}
  recDt = datetime.now() if recDt is None else recDt
  action2cachedOptions = {} if action2cachedOptions is None else action2cachedOptions
  '''
  with open("BeforegetPossibleActionsCall.txt", "a") as text_file:
    text_file.write(str(userProfile)+'_'+str(hospital)+str(recDt) +'_' + str(execDt)+'_'+str(pushing) + "\n")
  '''
  # Get a list of planned actions we should check for this user.
  department = userProfile.department
  ''' 
  if (userProfile.user.username).lower() in ['hossein' ,'rclaure', 'thuynguyen1', 'jeff', 'dscheinker', 'steven']:
    import pickle
    f=open('BeforegetPossibleActionsFxn_'+str(userProfile)+str(recDt)+'.pkl', 'wb')
    pickle.dump({'department': department, 'action2cachedOptions':action2cachedOptions, 'pushing': pushing},f)
    f.close()
  '''

  totalActions = getPossibleActions(hospital, department, pushing, execDt, recDt=recDt, userProfile=userProfile)
  '''
  if (userProfile.user.username).lower() in ['hossein' ,'rclaure', 'thuynguyen1', 'jeff', 'dscheinker', 'steven']:
    import pickle
    f=open('totalActions_'+str(userProfile)+str(recDt)+'.pkl', 'wb')
    pickle.dump(totalActions,f)
    f.close()
  
  with open("aftergetPossibleActionsCall.txt", "a") as text_file:
    text_file.write(str(userProfile)+'_'+str(hospital)+str(recDt) +'_' + str(execDt)+'_'+str(pushing) + "\n")
  '''
  # Run our decision loops to get our set of recommendations.
  specializedActions = [x for x in totalActions if x.isSpecialized]
  genericActions = [x for x in totalActions if not x.isSpecialized]
  specializedActions2options = getSpecializedOptions(userProfile, specializedActions, execDt, recDt,
                                                      action2cachedOptions=action2cachedOptions)
  '''
  import pickle
  if (userProfile.user.username).lower() in ['hossein' ,'rclaure', 'thuynguyen1', 'jeff', 'dscheinker', 'steven']:
    f=open('specializedActions2options'+str(userProfile)+str(recDt)+'.pkl', 'wb')
    pickle.dump(specializedActions2options,f)
    f.close()
  '''
  generalActions2options = getGeneralOptions(genericActions, hospital, execDt, recDt)

  # Update our action2option variable with our recommendations.
  action2options.update(specializedActions2options)
  action2options.update(generalActions2options)

  # Create a list from all the recommendations we have computed.
  allOptions = []
  for action in action2options:
    allOptions += action2options[action]
  '''
    if action.pk in [163, 165, 168]:
     logParams = {
      "event": event.RAN_DECISION_LOOP,
      "action.pk":action.pk,
      "specializedActions2options": str(len(specializedActions2options)),
      "generalActions2options":str(len(generalActions2options)),
      "userProfile": userProfile.user.username,
      "recDt":str(recDt),
      "pushing":str(pushing)
     }
     logger.info(**logParams)
  '''  
  

  # Finally filter these recommendations based on what we have sent recently.
  filteredOptions = filterOptionsWithLogs(userProfile, totalActions, allOptions, recDt, pushing)
  return filteredOptions, action2options


def filterOptionsWithLogs(userProfile, totalActions, allOptions, recDt, pushing):
  ret = allOptions

  if len(totalActions) > 0:
    longestCoolDown = max([x.coolDown for x in totalActions])
    optionIds = [x.id for x in allOptions if x.plannedAction.coolDown > 0]

    # Look for the oldest.
    logDicts = PlannedActionLog.objects.filter(userProfile=userProfile)\
      .filter(optionId__in=optionIds)\
      .filter(when__gte=recDt - timedelta(seconds=3600 * longestCoolDown))\
      .values('when', 'optionId', 'pushed', 'muteUntil')
    '''  
    import pickle
    store = {'optionIds':optionIds, 'PlannedActionLog':PlannedActionLog.objects.filter(userProfile=userProfile), 'longestCoolDown':longestCoolDown, 'logDicts':logDicts}
    if (userProfile.user.username).lower() in ['hossein' ,'rclaure', 'thuynguyen1', 'jeff', 'dscheinker', 'steven']:
      f=open('logDicts'+str(userProfile)+str(recDt)+'.pkl', 'wb')
      pickle.dump(store,f)
      f.close()
    '''   
    # If we are filtering for a web query, pretend they did not see the pushed
    # recommendations. If they are pushing, then we care about all things they
    # have seen (don't send same rec twice, etc.).
    optionId2latest = {}
    for logDict in logDicts:
      if not pushing and logDict['pushed']:  # If web query, don't worry about the pushed recommondations.
        continue
      elif logDict['optionId'] in optionId2latest:
        if logDict['when'] > optionId2latest[logDict['optionId']]['when']:
          optionId2latest[logDict['optionId']] = logDict
      else:
        optionId2latest[logDict['optionId']] = logDict

    ret = []
    for option in allOptions:
      if option.id in optionId2latest:
        latestLog = optionId2latest[option.id]
        if latestLog['when'] + timedelta(hours=option.plannedAction.coolDown) <= recDt\
            and (latestLog['muteUntil'] is None or latestLog['muteUntil'] < datetime.now()):
          ret.append(option)
      else:
        ret.append(option)

  return ret


# Get options from a list of planned specialized actions.
def getSpecializedOptions(userProfile, plannedActions, execDt, recDt, action2cachedOptions=None):
  ret = {}
  action2cachedOptions = {} if action2cachedOptions is None else action2cachedOptions
  import pickle

  for action in plannedActions:
    '''
    if action.pk in [163, 165, 168]:
      classType = action.getActionClass()
      decisionLoop = classType()
      f=open('getSpecializedOptions'+str(action.pk)+'_'+str(recDt)+'.pkl', 'wb')
      sto={'userProfile':userProfile.user.username, 'isaction':str(action in action2cachedOptions)}
      pickle.dump(sto, f)
      f.close()
      logParams = {
        "event": event.RAN_DECISION_LOOP,
        "userProfile":userProfile.user.username,
        "recDt":str(recDt),
        "action2cachedOptions":str(action in action2cachedOptions),
        "decisionLoop.HAS_PERSONALIZED_OPTIONS":str(decisionLoop.HAS_PERSONALIZED_OPTIONS),
        "decisionName": action.name
      }
      logger.info(**logParams)
    '''
    startTime = time.time()

    # Try to run this planned actions for this user.
    try:
      classType = action.getActionClass()
      decisionLoop = classType()

      # Check to see if we have cached results for this loop.
      if not decisionLoop.HAS_PERSONALIZED_OPTIONS and action in action2cachedOptions:
        options = action2cachedOptions[action]
      else:
        options = decisionLoop.getOptions(userProfile, action, execDt, recDt)
      ret[action] = options
    except:
      # To reduce the frequency of these alerts, logging this, so splunk can
      # send a scheduled alert, Instead of relatime alert. ENG-4584.
      logParams = {
        "username": userProfile.user.username,
        "event": event.DECISION_LOOP_ERROR,
        "msg": 'getOptions() for action %d failed: %s' % (action.pk, traceback.format_exc())
      }
      logger.error(**logParams)

    totalTime = time.time() - startTime  # seconds
    msg = 'Ran "%s (pk=%d)" for %s' % (action.name, action.pk, userProfile.user.username)
    logParams = {
      "event": event.RAN_DECISION_LOOP,
      "msg": msg,
      "duration": float(totalTime),
      "decisionName": action.name,
      "ret": str(len(ret))
    }
    logger.info(**logParams)

  return ret


# Get options from a list of generic planned actions.
def getGeneralOptions(plannedActions, hospital, execDt, recDt):
  templates = [x.impact for x in plannedActions]
  templateVals = evaluateTemplates(templates, hospital, execDt, recDt)
  return dict([[action, [ActionOption(action, action.description, templateVals[idx], recDt)]] for idx, action in enumerate(plannedActions)])


def getAvailableOptions(userProfile, hospital, workArea, execDt, recDt=None):
  if recDt == None:
    recDt = datetime.now()

  action2options = {}
  actions = getPossibleActions(hospital, workArea, False, execDt, recDt=recDt)
  specializedActions = [x for x in actions if x.isSpecialized]
#  genericActions = [x for x in actions if not x.isSpecialized]

  # Filter out the specific ones.
  specializedActions2options = getSpecializedOptions(userProfile, specializedActions, execDt, recDt)

  # For generic ones, call evaluateTemplates(). Done this way to ensure
  # our ability to create forecasts together (covariance matrix, etc.)
  # sometime in the future.
#  generalActions2options = getGeneralOptions(genericActions, hospital, execDt, recDt)

  action2options.update(specializedActions2options)
#  action2options.update(generalActions2options)

  allOptions = []
  for action in action2options:
    allOptions += action2options[action]

  return allOptions


# Get a list of PlannedActions 
def getPossibleActions(hospital, workArea, pushing, execDt, recDt=None, userProfile=None):
  if recDt == None:
    recDt = datetime.now()
  '''
  with open("start_getPossibleActionsfxn.txt", "a") as text_file:
    text_file.write('start_getPossibleActionsfxn_'+str(hospital)+'_'+str(workArea)+'_'+str(pushing)+'_'+str(recDt)+'_'+str(userProfile) + "\n")
  '''
  delta = (execDt - recDt)
  hoursTilExec = (delta.days * 24 * 60 * 60 + delta.seconds) / (60 * 60.0)

  # Get PlannedActions from db.
  deptActions = PlannedAction.objects.filter(hospital=hospital)\
      .filter(workArea=workArea)\
      .filter(horizonMin__lte=hoursTilExec)\
      .filter(horizonMax__gte=hoursTilExec)\
      .filter(entireDept=True)\
      .all()

  if userProfile != None:
    mems = getMembershipForUserProfile(userProfile)
    memActions = [mem.action for mem in mems if mem.action.hospital==hospital]
    allActions = list(set(memActions + list(deptActions)))  # Remove duplicates.
  else:
    allActions = deptActions

  # Go through each of these and test the constraint.
  ret = []
  for action in allActions:
    if actionPassesConstraint(action, recDt, execDt, pushing):
      ret.append(action)

  return ret


# Have refrained from putting in PlannedAction since template logic is here.
def actionPassesConstraint(action, recDt, execDt, pushing=False):
  '''
  if action.pk in [163, 165, 168]:
   with open("logCLE2.txt", "a") as text_file:
    text_file.write('actionPassesConstraint'+str(action.pk)+ '_'+str(recDt) + str(action.pushConstraint)+str(pushing) + "\n")
  '''
  constraint = action.pushConstraint if pushing else action.constraint
  hospital = action.hospital
  return True if evaluateSingleTemplate(constraint, hospital, execDt, recDt, lastRan=action.lastRan) else False


def evaluateSingleTemplate(template, hospital, execDt, recDt=None, lastRan=None):
  return evaluateTemplates([template], hospital, execDt, recDt=recDt, lastRan=lastRan)[0]


# lastRan is a variable that can be used in templates.
def evaluateTemplates(templates, hospital, execDt, recDt=None, lastRan=None):
  if recDt == None:
    recDt = datetime.now()

  # Preprocess all the templates (constants, metrics, etc).
  templateVars = []
  for template in templates:
    processedTemplate, variableStrs = preprocessTemplate(template, hospital, execDt, recDt)
    templateVars.append([processedTemplate, variableStrs])

  # Go though and collect forecasts I will have to run.
  tagStrs = []
  for processedTemplate, variableStrs in templateVars:
    tagStrs += variableStrs
  str2values = getTemplateExpectedAndResid(tagStrs, hospital, execDt, recDt)

  # Finally for each template, produce sample values.
  ret = []
  for processedTemplate, variableStrs in templateVars:
    if len(variableStrs) == 0:  # No simulations needed.
      ret.append(eval(processedTemplate))
    else:
      templateValues = getSimulatedTemplateVals(processedTemplate, variableStrs, str2values, execDt, recDt)
      avgValue = sum(templateValues) / float(len(templateValues))
      ret.append(avgValue)

  return ret


def getSimulatedTemplateVals(procTemplate, variableStrs, str2values, execDt, recDt):
  numberOfSimulations = 100
  maxRanSize = 10000
  ret = []

  # For each simulation, get a residual number. That way, if residuals match up
  # (ie residual 1 is jan2 for all forcasts, residual 2 is ... etc), then our
  # simulated residuals will have good covariance with each other.
  # TODO: think about this more. Distributions / covar matrices / etc.
  for i in range(0, numberOfSimulations):
    templateToEval = procTemplate
    residualNum = random.randint(0, maxRanSize)

    # TODO: This could be an issue if string isnt unique.
    for variableStr in variableStrs:
      forecastVars = str2values[variableStr]
      lenResiduals = len(forecastVars['residuals'])
      if variableStr.startswith('residual'):
        replaceStr = forecastVars['residuals'][residualNum % lenResiduals]
      else:
        replaceStr = forecastVars['expected']
      templateToEval = templateToEval.replace(variableStr, str(replaceStr))

    ret.append(eval(templateToEval))

  return ret


def getTemplateExpectedAndResid(tagStrs, hospital, execDt, recDt):
  ret, name2result = {}, {}

  for tagStr in tagStrs:
    wsIndex = tagStr.strip().find(' ')
    tagName, argStr = tagStr[:wsIndex], tagStr[wsIndex:]
    args = parseTagArguments(argStr)

    assert (tagName in ['forecast', 'residual'])

    name = args['name']
    if not name in name2result:
      expected, residuals = getForecastExpectedAndResiduals(args, hospital, execDt, recDt)
      name2result[tagStr] = {
        'expected': expected,
        'residuals': residuals
      }
      ret[tagStr] = name2result[tagStr]

  return ret


# Take in the arguements for a forecast/residual and return the expected value
# in addition to the last X residuals.
# - TODO: add support for hourly metrics, etc.
def getForecastExpectedAndResiduals(args, hospital, execDt, recDt):
  tmpColumnName = 'value'
  name = args['name']
  modelType = args['type'] if args.get('type') != None else 'metric'
  forecast = getForecastForAction(name, modelType, hospital, execDt, recDt)

  trainStart = recDt - timedelta(days=365 * 2)
  forecastValues = forecast.getYValues(trainStart, recDt)
  dt2row = dict([(datetime(dt.year, dt.month, dt.day), [value,]) for dt, value in forecastValues])

  metricDelta = forecastValues[1][0] - forecastValues[0][0]  # TODO fix this.
  delta = execDt - recDt

  deltaSeconds = delta.days * 24 * 60 * 60 + delta.seconds
  metricSeconds = metricDelta.days * 24 * 60 * 60 + metricDelta.seconds

  trainStart = datetime(trainStart.year, trainStart.month, trainStart.day)
  trainEnd = datetime(recDt.year, recDt.month, recDt.day)
  config = {
    'interval': metricDelta,
    'constantInterval': metricDelta,
    'horizonSize': deltaSeconds / metricSeconds, # since block size is one.
    'blockSize': 1,
    'trainStart': str(trainStart),
    'trainEnd': str(trainEnd),
    'testEnd': str(trainEnd)
  }

  forecastConfig = {}
  forecastConfig.update(forecast.modelOptions)
  forecastConfig.update(config)

#  train, test = iterateThroughForecast(forecastConfig, [tmpColumnName], dt2row, verbose=False)
#  residuals = [x[2] - x[3] for x in train[tmpColumnName]]
#  expected = test[tmpColumnName][-1][2]
#
#  if isinstance(expected, np.ndarray):
#    expected = expected[0]

  return [0], [0]#expected, residuals


def getForecastForAction(name, modelType, hospital, execDt, recDt):
  delta = execDt - recDt
  deltaSeconds = delta.days * 24 * 60 * 60 + delta.seconds
  deltaHours = deltaSeconds / (60.0 * 60)

  query = Forecast.objects.filter(modelType=modelType)\
      .filter(horizon__lte=deltaHours)
  if modelType == 'metric':
    query = query.filter((Q(metricDef__metricName=name) & Q(metricDef__hospital=hospital)) | Q(name=name))
  else:
    query = query.filter(name=name)

  # Information on the Forecasts we found.
  forecasts = query.all()
  numForecasts = len(forecasts)

  # Look through forecasts for an appropriate one for this timeframe.
  reqForecast = None
  for forecast in query.all():
    if forecast.horizon + forecast.windowSize > deltaHours:
      if reqForecast != None:
        raise KeyError('Multiple forecasts found (name=%s)' % (str(name)))
      else:
        reqForecast = forecast

  # Raise a special exception if we don't find a good forecast for deltaHours.
  if numForecasts > 0 and reqForecast == None:
    raise KeyError('No forecast for distance %.2f hours (name=%s)' % (deltaHours, str(name)))

  # Otherwise raise a generic not found.
  if reqForecast == None:
    raise KeyError('Could not find forecast (name=%s)' % (str(name)))

  return reqForecast


def preprocessTemplate(template, hospital, execDt, recDt=None):
  if recDt == None:
    recDt = datetime.now()

  variableStrings = []
  if template.find('/>') >= 0:
    evalStr = ""
    nodes = template.split('/>')
    for node in nodes:
      nodeString = ""
      inQuotes = False
      for i in range(len(node) - 1, -1, -1):  # Iterate backwards.
        if node[i] == '"' or node[i] == "'":
          inQuotes = not inQuotes
        elif node[i] == '<' and not inQuotes:
          tagStr = node[i:].strip()
          if tagStr.startswith('<forecast') or tagStr.startswith('<residual'):
            variableStrings.append(tagStr[1:])
            nodeString = node[:i] + tagStr[1:]  # We will manually string replace forecasts, residuals, etc.
          else:
            nodeString = node[:i] + getTagValue(tagStr, hospital, execDt, recDt)
          break
      if nodeString == "":
        nodeString += node
      evalStr += nodeString
  else:
    evalStr = template

  return evalStr, variableStrings


# ' name="zzz" blah= "foo"' -> {name: zzz, blah: foo}
def parseTagArguments(argsStr):
  ret = {}
  #args = argStr.split(' ')
  argsStr = argsStr.strip()

  args = []
  inQuotes = False

  argStr = ''
  seenEqual, seenValue = False, False
  for i in range(len(argsStr)):
    if not inQuotes and seenEqual and seenValue and argsStr[i] == ' ':
      if len(argStr) > 0:
        args.append(argStr)
      argStr = ''
      seenEqual, seenValue = False, False
    elif argsStr[i] == '"' or argsStr[i] == "'":
      inQuotes = not inQuotes
      argStr += argsStr[i]
      if not inQuotes and not seenValue and seenEqual:
        seenValue = True
    else:
      if not inQuotes and not seenEqual and argsStr[i] == '=':
        seenEqual = True
      if not inQuotes and not seenValue and seenEqual and argsStr[i] == ' ':
        seenValue = True
      argStr += argsStr[i]
  if len(argStr) > 0:
    args.append(argStr)

  for arg in args:
    argTuple = arg.strip().split('=')
    if len(argTuple) == 2:
      key, value = argTuple
      ret[key.strip()] = value.strip(" '\"")

  #print '   ', argsStr, '->', ret
  return ret


def getTagValue(tag, hospital, execDt, recDt):
  tag = tag.strip()
  if tag.startswith('<'):
    tag = tag[1:].strip()

  wsIndex = tag.find(' ')
  if wsIndex >= 0:
    tagName, argStr = tag[:wsIndex], tag[wsIndex:]
    args = parseTagArguments(argStr)

    if tagName == 'constant':
      ret = str(getConstantValue(args, hospital, execDt, recDt))
    elif tagName == 'metric':
      ret = str(getMetricValue(args, hospital, execDt, recDt))
    elif tagName == 'residual' or tagName == 'forecast':
      ret = tag
    else:
      raise NotImplementedError('tag name "%s" not supported' % tagName)
  else:
    # dt, etc.
    pass

  return str(ret)


def getConstantValue(args, hospital, execDt, recDt):
  name = args.get('name')
  constInstance = ConstantInstance.objects.filter(constant__name=name)\
    .filter(hospital=hospital)\
    .filter(fromDt__lte=execDt)\
    .filter(toDt__gte=execDt).all()

  if len(constInstance) == 1:
    return constInstance[0].value
  elif len(constInstance) > 1:
    raise KeyError('more than 1 constant with args "%s" at hospital "%s" found' % (str(args), str(hospital)))
  else:
    raise KeyError('constant with args "%s" at hospital "%s" not found' % (str(args), str(hospital)))


def getMetricValue(args, hospital, execDt, recDt):
  name = args.get('name')
  metricDefs = MetricDefinition.objects.filter(metric__name=name)\
    .filter(hospital=hospital).all()

  if len(metricDefs) == 1:
    metricDef = metricDefs[0]

    ret = None
    if args.get('target') != None:
      ret = metricDef.target
    elif args.get('baseline') != None:
      ret = metricDef.baseline
    else:
      dt = recDt if args.get('dt') != 'execDt' else execDt
      smear = int(args.get('smear')) - 1 if args.get('smear') != None else 0
      lag = int(args.get('lag')) if args.get('lag') != None else 0

      delta = timedelta(seconds=24 * 60 * 60)
      end = dt + delta - timedelta(seconds=1)
      start = dt - smear * delta
      start, end = start - lag * delta, end - lag * delta

      ret = singleMetric.getHistoricalAverage(metricDef, start, end)
    return ret
  elif len(metricDefs) > 1:
    raise KeyError('more than 1 metricDef with args "%s" at hospital "%s" found' % (str(args), str(hospital)))
  else:
    raise KeyError('metricDef with args "%s" at hospital "%s" not found' % (str(args), str(hospital)))
