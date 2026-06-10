from asyncio import Condition
from lib2to3.pgen2.driver import load_grammar
from re import A
import Classes.custom_menu_bar_class
import matplotlib.pyplot as plt
import importlib
import qtm
import numpy as np
import subprocess
import time
import os
import csv
import sys
import inspect
import pandas as pd
import math
import ast
from datetime import datetime
# When using 32-bit Python310 supplied with QTM 2023.3(build 12577), pip cannot provide compatible compilers to build pandas, sklearn, and scipy. 
# While meson-python and ninja could be installed, a cython compiler is not 32-bit compatible.
# The packages however could be installed from "wheel" binaries (those that could not be built due to 32-bit-incompatible compilers)
# https://www.lfd.uci.edu/~gohlke/pythonlibs/ has these wheels! Maybe slightly outdated packages, but seem to work after manual installation of each one via pip.
import pandas as pd 
#import sklearn 
import scipy
#from scipy.linalg import lstsq

import helpers.menu_tools
import helpers.traj

importlib.reload(helpers.menu_tools)
importlib.reload(helpers.traj)

from helpers.menu_tools import add_menu_item, add_command
from helpers.traj import get_default_markerset_marker, get_selected_markerset_marker

this_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if this_dir not in sys.path:
    sys.path.append(this_dir)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ////////   P R I V A T E   F U N C T I O N S   ////////
# - - - - - - - - - - - - - - - - - - - - - - - - - - -
# region [ COLLAPSE / EXPAND ]
# def _reload_script_modules():
#     # Python's default behaviour is to cache imported scripts. This
#     # means that changes you make to these scripts will not show up in
#     # QTM despite pressing the "Reload scripts" button. However, running
#     # "importlib.reload()" on these scripts will force Python to reload them.
#     importlib.reload(Classes.custom_menu_bar_class)
# endregion


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ////////   E X P O R T E D   F U N C T I O N S   ////////
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# # region [ COLLAPSE / EXPAND ]
# def setup_basic():
#     custom_menu_bar_instance.setup_menu_basic()


# def setup_advanced():
#     custom_menu_bar_instance.setup_menu_advanced()

# def delete():
#     custom_menu_bar_instance.delete_menu()
# # endregion


# AK functions

def f(N):
	gtf(N)

def gtf(N): # Go To Frame
	# Using a hotkey to prompt for a frame number input and navigate there
	#FrameN = input("Enter frame number: ")
	NFrames = qtm.gui.timeline.get_frame_count()
	N = int(N)
	if (N < 1) | (N > (NFrames)):
		print("Frame out of range. Aborting...")
	else:
		qtm.gui.timeline.set_current_frame(N-1)

def Whip_Names_to_W_Names():
	seriesIDs = qtm.data.series._3d.get_series_ids()
	fixedCnt = 0
	for id in seriesIDs:
		fullname = qtm.data.object.trajectory.get_label(id)
		if fullname is not None:
			if (len(fullname) > 3) & (not ("Target" in fullname)):
				fixedName = "W_" + fullname
				qtm.data.object.trajectory.set_label(id,fixedName)
				fixedCnt = fixedCnt + 1
	if fixedCnt > 0:
		print(f"Added prefix W_ to {fixedCnt} marker names")

def Whip_W_Names_to_Names():
	seriesIDs = qtm.data.series._3d.get_series_ids()
	fixedCnt = 0
	for id in seriesIDs:
		fullname = qtm.data.object.trajectory.get_label(id)
		if fullname is not None:
			if (len(fullname) > 3) & (not ("Target" in fullname)):
				fixedName = fullname.split("_")[1]
				qtm.data.object.trajectory.set_label(id,fixedName)
				fixedCnt = fixedCnt + 1
	if fixedCnt > 0:
		print(f"Removed prefix W_ from {fixedCnt} marker names")


# 2023.2 says the method has not been implemented yet.
def WhipTrimBits():
	NFrames = qtm.gui.timeline.get_frame_count()
	if NFrames > 5000:
		qtm.gui.timeline.set_measured_range({"start": 5, "end": (NFrames-5)})
		# Ready to t
	else:
		print(f'The recording seems too short and is probably corrupt. Check manually.')
	qtm.gui.send_command('load_label_list')
	

def MarkerRemoveVirtualParts():
	NFrames = qtm.gui.timeline.get_frame_count()
	s = qtm.gui.selection.get_selections()
	if s == None:
		print('Nothing is selected')
		return
	if len(s) == 0:
		print('Nothing is selected')
		return
	if len(s) > 1:
		print('Too many objects were selected. Aborting..')
		return
	if not ('trajectory' in s[0]['type']):
		print('Something that is not a trajectory was selected. Aborting..')
		return
	idsel = s[0]["id"]
	idname = qtm.data.object.trajectory.get_label(idsel)
	PartsAll = qtm.data.object.trajectory.get_parts(idsel)
	IndVirtParts = []
	for ipart,part in enumerate(PartsAll):
		if part['type'] == 'virtual':
			IndVirtParts.append(ipart)
	#for ipart in np.flip(IndVirtParts):
	if len(IndVirtParts) > 0:
		try:
			qtm.data.object.trajectory.delete_parts(idsel,IndVirtParts)
			print(f'Removed virtual parts {IndVirtParts} from {idname}')
		except Exception as e:
			print(f'Could not remove Parts {IndVirtParts} from {idname},exception {e}')
	else:
		print('No virtual parts')
	


# When training, update them. 
def updMeanSD(Nold,MeanOld,SDold,NewSamp):
	if np.any(np.isnan(NewSamp)):
		NewMean = MeanOld
		NewSD = SDold
		Nnew = Nold
		return NewMean, NewSD
	Nnew = Nold + 1
	if np.any(np.isnan(MeanOld)):
		NewMean = NewSamp
	else:
		NewMean = (MeanOld * Nold + NewSamp) / Nnew
	OldVar = SDold**2
	if np.any(np.isnan(OldVar)):
		NewSD = np.full(NewSamp.shape, 0)
	else:
		NewVar = (OldVar * (Nold - 1) + (NewSamp - NewMean)**2) / (Nnew - 1)
		NewSD = np.sqrt(NewVar)
	if ~np.all(np.isnan(NewMean)):
		NewMean = np.round(NewMean,0)
	if ~np.all(np.isnan(NewSD)):
		NewSD = np.round(NewSD,0)
	return NewMean,NewSD


# Label everything in the current frame (around the first frame, using the posture with the whip pointing forward)
def LabelAllInFrame_Train():
	# Get all trajectories in the frame +-
	# Get all markernames

	NFrames = qtm.gui.timeline.get_frame_count()
	iCur = qtm.gui.timeline.get_current_frame()

	irange = np.arange(iCur - 40, iCur + 40)
	irange = irange[(irange >= 0) & (irange < NFrames)]

	ProjDir = qtm.settings.directory.get_project_directory()
	MarkerNames = []
	file_path = f'{ProjDir}/WholeBody&Whip&Target.txt'
	if os.path.exists(file_path):
		with open(file_path,'r') as file:
			for line in file:
				if '<Name>' in line:
					S = line.split('<Name>')[1].split('</Name>')[0]
					MarkerNames.append(S)
	#print(MarkerNames)
	#print(irange)
	DataAll = {}
	DataPose = {}
	for iname,name in enumerate(MarkerNames):
		idt = qtm.data.object.trajectory.find_trajectory(name)
		if idt == None:
			continue
		#X = GetMarkerDataAll(idt)
		#print(irange[0])
		X = qtm.data.series._3d.get_samples(idt, {"start":irange[0].item(), "end":irange[-1].item()})
		#print(X)
		#print(iname, name)
		#print(X)
		Y = np.full((len(X),3),np.nan)
		for i in range(len(X)):
			if X[i] == None:
				continue
			Y[i,:] = (X[i]['position'])
		X = Y.round(0)
		#print(name)
		#print(X)
		
		DataAll[name] = X
		DataPose[name] = np.nanmean(X,axis=0)
		#print(f'{name}: {X} - {np.nanmean(X,axis=0)}')
	# Get distances: at the pose and general constraints over time
	#return

	DistAllBounds = np.full((len(MarkerNames),len(MarkerNames),2),np.nan)
	DistAllPose = np.full((len(MarkerNames),len(MarkerNames)),np.nan)
	for iname1,name1 in enumerate(MarkerNames):
		for iname2,name2 in enumerate(MarkerNames):
			if iname2 <= iname1:
				continue # Only above the diagonal, to save half of computation time.
			Dist = np.linalg.norm(DataAll[name1] - DataAll[name2], axis=1)
			DistPose = np.nanmean(Dist)
			#print(f'{name1}-{name2}: {DistPose}')
			DistBounds = np.nanpercentile(Dist, [1, 99])
			DistAllBounds[iname1, iname2 , :] = DistBounds
			DistAllBounds[iname2, iname1 , :] = DistBounds
			DistAllPose[iname1, iname2] = DistPose
			DistAllPose[iname2, iname1] = DistPose
	#return

	DataPosN = np.full((len(MarkerNames),1),0)
	DataPosM = np.full((len(DataPose),3),np.nan)
	DataPosS = np.full((len(DataPose),3),np.nan)
	DataDistN = np.full((len(MarkerNames),len(MarkerNames)),0)
	DataDistM = np.full((len(DataPose),len(DataPose)),np.nan)
	DataDistS = np.full((len(DataPose),len(DataPose)),np.nan)
	DataDistBoundsM = np.full((len(DataPose),len(DataPose),2),np.nan)
	DataDistBoundsS = np.full((len(DataPose),len(DataPose),2),np.nan)


	# Read from the file.
	file_path = f'{ProjDir}/!LabelFirstFrameModel.txt'
	if os.path.exists(file_path):
		total_lines = sum(1 for _ in open(file_path))
		#print('rows',total_lines)
		with open(file_path,'r') as file:
			csv_readobj = csv.reader(file, delimiter=',')
			flag_ReadingPos = False
			lastrow = 0
			#print("AA")
			for irow,row in enumerate(csv_readobj):
				#print(irow)
				if 'Marker indices, names, and positions' in row[0]:
					flag_ReadingPos = True
					continue
				if 'Symmetric matrix of distance means' in row[0]:
					flag_ReadingPos = False
					#read matrix
					lastrow = irow
					break # remember the irow
				if flag_ReadingPos:
					parts = row[0].split(' - ')
					#print(parts)
					iname = int(parts[0])
					name = parts[1].strip()
					#print(parts[2].strip('[]'))
					DataPosM[iname,:] = np.fromstring(parts[2].strip('[]'), dtype=float,sep=' ')
					DataPosS[iname,:] = np.fromstring(parts[3].strip('[]'), dtype=float,sep=' ')
					#print(parts[4].strip('[]'))
					DataPosN[iname] = int(parts[4].strip('[]'))
			
			#print(row)
			shead = lastrow+1
			DataDistM = np.genfromtxt(file_path, delimiter=',', skip_header=shead, skip_footer=total_lines-shead-len(MarkerNames), usecols=np.arange(len(MarkerNames)))
			DataDistS = np.genfromtxt(file_path, delimiter=',', skip_header=(shead + 1 + len(MarkerNames)), skip_footer = total_lines - shead - 1 - 2*len(MarkerNames), usecols=np.arange(len(MarkerNames)))
			DataDistN = np.genfromtxt(file_path, delimiter=',', skip_header=(shead + 2 + 2*len(MarkerNames)),  usecols=np.arange(len(MarkerNames)))

	else:
		print(f'CSV with customized thresholds not found. Run VerifyDist() once to create one, then use _Get() to display them or _Set(e.g. "w1-W_RWristOut",150) to change a value for a pair of markers')
		


		
		

	for iname1,name1 in enumerate(MarkerNames):
		#print(f'{DataPose[name1]}')
		M, S = updMeanSD(DataPosN[iname1],DataPosM[iname1,:],DataPosS[iname1,:],DataPose[name1])
		#print(f'{name1}: {M} - {S}')
		DataPosM[iname1,:] = M
		DataPosS[iname1,:] = S
		if ~np.any(np.isnan(DataPose[name1])):
			DataPosN[iname1] += 1
		for iname2,name2 in enumerate(MarkerNames):
			if iname2 <= iname1:
				continue
			M, S = updMeanSD(DataDistN[iname1,iname2],DataDistM[iname1,iname2],DataDistS[iname1,iname2],DistAllPose[iname1,iname2])
			#if len(M) > 1:
			#print(f'{name1}, {name2}, {M}, {S}')
			DataDistM[iname1,iname2] = M
			DataDistS[iname1,iname2] = S
			DataDistM[iname2,iname1] = M
			DataDistS[iname2,iname1] = S
			M, S = updMeanSD(DataDistN[iname1,iname2],DataDistBoundsM[iname1,iname2,:],DataDistBoundsS[iname1,iname2,:],DistAllBounds[iname1, iname2 , :])
			DataDistBoundsM[iname1,iname2,:] = M
			DataDistBoundsS[iname1,iname2,:] = S
			DataDistBoundsM[iname2,iname1,:] = M
			DataDistBoundsS[iname2,iname1,:] = S
			if ~np.any(np.isnan(DistAllPose[iname1,iname2])):
				DataDistN[iname1,iname2] += 1
				DataDistN[iname2,iname1] += 1


	#return

	# Write to the file mean, SD and Nsamples for each parameter
	DTcurr = datetime.now()
	with open(file_path,'w',newline='') as file:
		csv_writer = csv.writer(file,delimiter=',') # Write the file
		csv_writer.writerow(['***Marker positions and inter-marker differences for autolabeling the "whip forward" frame. Last edit time: ', DTcurr])
		#csv_writer.writerow(f'DataPosN = {DataPosN + 1}')
		csv_writer.writerow(['***Marker indices, names, and positions: mean - sd - N'])
		for iname,name in enumerate(MarkerNames):
			csv_writer.writerow([f'{iname} - {name} - {DataPosM[iname]} - {DataPosS[iname]} - {DataPosN[iname]}'])
		csv_writer.writerow(['***Symmetric matrix of distance means'])
		for irow,row in enumerate(DataDistM):
			csv_writer.writerow(row.astype(int))
		csv_writer.writerow(['***Symmetric matrix of distance SDs'])
		for irow,row in enumerate(DataDistS):
			csv_writer.writerow(row.astype(int))
		csv_writer.writerow(['***Symmetric matrix of distance N'])
		for irow,row in enumerate(DataDistN):
			csv_writer.writerow(row.astype(int))
	print(f'Updated the model with the Position and Inter-marker distance data from the current frame and a few adjacent ones')


def MVN_pdf(x,xm,xsd):
		S = np.eye(3)
		S[0,0] = xsd[0] + 0.1 * max(xsd)
		S[1,1] = xsd[1] + 0.1 * max(xsd)
		S[2,2] = xsd[2] + 0.1 * max(xsd)
		if max(xsd) < 0.01:
			S = 0.1 * np.eye(3)
		S = S * 10
		f = 1/(2*np.pi) **(3/2) / np.sqrt(np.linalg.det(S)) * np.exp(-1/2 * (x - xm).T @ np.linalg.inv(S) @ (x-xm))
		#f = 
		return f
def UV_pdf(x,xm,sd):
	f = 1/np.sqrt(2*np.pi) / np.sqrt(sd) * np.exp(-1/2 * (x-xm)**2 / sd**2)
	return f

def LabelAllInFrame_Label():
	# Get all trajectories in the frame +-
	# Get all markernames

	NFrames = qtm.gui.timeline.get_frame_count()
	iCur = qtm.gui.timeline.get_current_frame()

	irange = np.arange(iCur - 40, iCur + 40)
	irange = irange[(irange >= 0) & (irange < NFrames)]

	ProjDir = qtm.settings.directory.get_project_directory()
	MarkerNames = []
	file_path = f'{ProjDir}/WholeBody&Whip&Target.txt'
	if os.path.exists(file_path):
		with open(file_path,'r') as file:
			for line in file:
				if '<Name>' in line:
					S = line.split('<Name>')[1].split('</Name>')[0]
					MarkerNames.append(S)
	#print(MarkerNames)
	#print(irange)
					
	DataPosN = np.full((len(MarkerNames),1),0)
	DataPosM = np.full((len(MarkerNames),3),np.nan)
	DataPosS = np.full((len(MarkerNames),3),np.nan)
	DataDistN = np.full((len(MarkerNames),len(MarkerNames)),0)
	DataDistM = np.full((len(MarkerNames),len(MarkerNames)),np.nan)
	DataDistS = np.full((len(MarkerNames),len(MarkerNames)),np.nan)
	# Read from the file.
	file_path = f'{ProjDir}/!LabelFirstFrameModel.txt'
	print(f"Reading model from the {file_path}")
	if os.path.exists(file_path):
		total_lines = sum(1 for _ in open(file_path))
		#print('rows',total_lines)
		with open(file_path,'r') as file:
			csv_readobj = csv.reader(file, delimiter=',')
			flag_ReadingPos = False
			lastrow = 0
			#print("AA")
			for irow,row in enumerate(csv_readobj):
				#print(irow)
				if 'Marker indices, names, and positions' in row[0]:
					flag_ReadingPos = True
					continue
				if 'Symmetric matrix of distance means' in row[0]:
					flag_ReadingPos = False
					#read matrix
					lastrow = irow
					break # remember the irow
				if flag_ReadingPos:
					parts = row[0].split(' - ')
					#print(parts)
					iname = int(parts[0])
					name = parts[1].strip()
					#print(parts[2].strip('[]'))
					DataPosM[iname,:] = np.fromstring(parts[2].strip('[]'), dtype=float,sep=' ')
					DataPosS[iname,:] = np.fromstring(parts[3].strip('[]'), dtype=float,sep=' ')
					#print(parts[4].strip('[]'))
					DataPosN[iname] = int(parts[4].strip('[]'))
			
			#print(row)
			shead = lastrow+1
			DataDistM = np.genfromtxt(file_path, delimiter=',', skip_header=shead, skip_footer=total_lines-shead-len(MarkerNames), usecols=np.arange(len(MarkerNames)))
			DataDistS = np.genfromtxt(file_path, delimiter=',', skip_header=(shead + 1 + len(MarkerNames)), skip_footer = total_lines - shead - 1 - 2*len(MarkerNames), usecols=np.arange(len(MarkerNames)))
			DataDistN = np.genfromtxt(file_path, delimiter=',', skip_header=(shead + 2 + 2*len(MarkerNames)),  usecols=np.arange(len(MarkerNames)))
	else:
		print(f'CSV with customized thresholds not found. Do not delete that file... Open the "whip forward" frame in a few labeled records and run LabelAllInFrame_Train()')



	# Get data
	IDsUnind = []
	IDsAll = qtm.data.series._3d.get_series_ids()
	DataAll = np.full((len(IDsAll),3),np.nan)
	jid = 0
	for iid, id in enumerate(IDsAll):
		#print(id)
		X = qtm.data.series._3d.get_samples(id, {"start":irange[0].item(), "end":irange[-1].item()})
		Y = np.full((len(X),3),np.nan)
		for i in range(len(X)):
			if X[i] == None:
				continue
			Y[i,:] = (X[i]['position'])
		if (not np.all(np.isnan(Y))) & (not np.all(np.isnan(np.nanmean(Y)))):
			IDsUnind.append(id)
			DataAll[jid,:] = (np.nanmean(Y,axis=0))
			jid += 1
	print(f'Unidentified trajectories in this frame: {len(IDsUnind)}')
	DataAll = DataAll[:len(IDsUnind),:]
	#print(DataAll)
	#print(IDsUnind)
	# Find distances
	DistData = np.full((len(DataAll),len(DataAll)),np.nan)
	for i in range(len(DataAll)):
		for j in range(len(DataAll)):
			if j <= i:
				continue # Only above the diagonal, to save half of computation time.
			DistData[i,j] = np.linalg.norm(DataAll[i,:] - DataAll[j,:])
			DistData[j,i] = DistData[i,j]

	#print('Distances:')
	#print(DistData)
	
	# Now the fun part. How to assign?
	LabelsID = {}
	for iname,name in enumerate(MarkerNames):
		idt = qtm.data.object.trajectory.find_trajectory(name)
		LabelsID[name] = (idt)
	#print(IDsUnind[0])
	#print(LabelsID['w1'])
		
	# Now, decision making. 
	


	# Start with Positional metric. Use PDF, assuming 3D Gaussian...
	ScorePos = np.full((len(DataAll),len(MarkerNames)), np.nan)
	NClosest = 5
	DistModData = np.full((len(DataAll),len(MarkerNames)),np.nan)
	for irow in range(len(DataAll)):
		rowData = DataAll[irow,:]
		for irowMod in range(len(MarkerNames)):
			rowModelM = DataPosM[irowMod,:]
			rowModelS = DataPosS[irowMod,:]
			ScorePos[irow,irowMod] = MVN_pdf(rowData,rowModelM,rowModelS)
	
	for irow in range(len(DataAll)):
		row = ScorePos[irow,:]
		idas = np.argsort(row)[::-1]
		rs = [row[i] for i in idas[:NClosest]]
		mnames = [MarkerNames[i] for i in idas[:NClosest]]
		#print(f'{mnames};  {rs}')
	

	# And add distances.
	# NClosest = 5 # These distances alone are not sufficient. 
	# for irow in range(len(DataAll)):
	# 	rowData = DistData[irow,:]
	# 	rowData = rowData[abs(rowData) < 10000]
	# 	rowData = np.sort(rowData[~np.isnan(rowData)])
	# 	#rowData = rowData[:NClosest]
	# 	#print(rowData)
	# 	SS = np.full((len(MarkerNames)),np.nan)
	# 	SSMean = np.full((len(MarkerNames)),np.nan)
	# 	for irowMod in range(len(MarkerNames)):
	# 		rowModelM = DataDistM[irowMod,:]
	# 		rowModelM = rowModelM[abs(rowModelM) < 10000]
	# 		rowModelM = np.sort(rowModel[~np.isnan(rowModel)])
	# 		#rowModel = rowModel[:NClosest]
	# 		#print(rowModel)
	# 		SS[irowMod] = np.sqrt(np.nanmean((rowData - rowModel)**2)).astype(int)
	# 		SSMean[irowMod] = np.sqrt(((np.nanmean(rowData) - np.nanmean(rowModel))**2)).astype(int)
	# 		#print(f'{MarkerNames[irowMod]}: {SS[irowMod]}. Grand mean square diff: {SSMean[irowMod]}') # seems ok
		
	# 	Score[irow,:] = SS / np.max(SS)
	# 	# print(SS)
	# 	idxasc = np.argsort(SS)
	# 	# print(idxasc)
	# 	SS = np.sort(SS)
	# 	for i in range(5):
	# 		print(f'{MarkerNames[idxasc[i]]}: {SS[i]}') #Grand mean square diff: {SSMean[idxasc[i]]}

	# 	imatch = np.argmin(SS)
	# 	#print(f'Matched with {MarkerNames[idxasc[imatch]]}')
	# for irow in range(len(DataAll)):
	# 	a = Score[irow,:]
	# 	idxasc = np.argsort(a)
	# 	mnames = [MarkerNames[i] for i in idxasc[:NClosest]]
	# 	scores = [np.round(Score[irow,i],3) for i in idxasc[:NClosest]]
	# 	print(f'{IDsUnind[irow]} is a closest match to {mnames} with scores {scores}')

	
	# In this file (SA_1_0004) the first to UNID are W_RHeelBack and W_WaistRFront

	#print(rowData)
		
	# Attempt assigning
		# Assignment itself should work like that: Get traj ID for the labels. Then qtm.data.object.trajectory.move_parts(id_from,id_to,[]) to move all parts.
	# for irow in range(len(DataAll)):
	# 	rowData = DataAll[irow,:]
	# 	for irowMod in range(len(MarkerNames)):
	# 		rowModelM = DataPosM[irowMod,:]
	# 		rowModelS = DataPosS[irowMod,:]
	# 		ScorePos[irow,irowMod] = MVN_pdf(rowData,rowModelM,rowModelS)
	
	LabelsM = np.zeros((len(DataAll),len(MarkerNames)))
	print('Using Distances to assign. ')
	for irow in range(len(DataAll)):
		row = ScorePos[irow,:]
		idas = np.argsort(row)[::-1]
		rs = [row[i] for i in idas[:NClosest]]
		mnames = [MarkerNames[i] for i in idas[:NClosest]]
		LabelsM[irow,idas[:NClosest]] = rs
		#print(f'{mnames};  {rs}')
		try:
			qtm.data.object.trajectory.move_parts(IDsUnind[irow],LabelsID[mnames[0]])
		except Exception as e:
			print(f'Could not label {IDsUnind[irow]} as {mnames[0]}')
	#print(LabelsM)
	

	return

		# Add some exponential weighting, to consider far markers less?


	DataAll
	DistData

	DataPosM
	DataDistM

	

	return
	
	
	
	


	# Get distances: at the pose and general constraints over time
	DistAllBounds = np.full((len(MarkerNames),len(MarkerNames),2),np.nan)
	DistAllPose = np.full((len(MarkerNames),len(MarkerNames)),np.nan)
	for iname1,name1 in enumerate(MarkerNames):
		for iname2,name2 in enumerate(MarkerNames):
			if iname2 <= iname1:
				continue # Only above the diagonal, to save half of computation time.
			Dist = np.linalg.norm(DataAll[name1] - DataAll[name2], axis=1)
			DistPose = np.nanmean(Dist)
			DistBounds = np.nanpercentile(Dist, [1, 99])
			DistAllBounds[iname1, iname2 , :] = DistBounds
			DistAllBounds[iname2, iname1 , :] = DistBounds
			DistAllPose[iname1, iname2] = DistPose
			DistAllPose[iname2, iname1] = DistPose


	
		


		
		

	for iname1,name1 in enumerate(MarkerNames):
		M, S = updMeanSD(DataPosN[iname1],DataPosM[iname1,:],DataPosS[iname1,:],DataPose[name1])
		DataPosM[iname1,:] = M
		DataPosS[iname1,:] = S
		if ~np.any(np.isnan(DataPose[name1])):
			DataPosN[iname1] += 1
		for iname2,name2 in enumerate(MarkerNames):
			if iname2 <= iname1:
				continue
			M, S = updMeanSD(DataDistN[iname1,iname2],DataDistM[iname1,iname2],DataDistS[iname1,iname2],DistAllPose[iname1,iname2])
			#if len(M) > 1:
			#print(f'{name1}, {name2}, {M}, {S}')
			DataDistM[iname1,iname2] = M
			DataDistS[iname1,iname2] = S
			DataDistM[iname2,iname1] = M
			DataDistS[iname2,iname1] = S
			M, S = updMeanSD(DataDistN[iname1,iname2],DataDistBoundsM[iname1,iname2,:],DataDistBoundsS[iname1,iname2,:],DistAllBounds[iname1, iname2 , :])
			DataDistBoundsM[iname1,iname2,:] = M
			DataDistBoundsS[iname1,iname2,:] = S
			DataDistBoundsM[iname2,iname1,:] = M
			DataDistBoundsS[iname2,iname1,:] = S
			if ~np.any(np.isnan(DistAllPose[iname1,iname2])):
				DataDistN[iname1,iname2] += 1
				DataDistN[iname2,iname1] += 1




	# Write to the file mean, SD and Nsamples for each parameter
	DTcurr = datetime.now()
	with open(file_path,'w',newline='') as file:
		csv_writer = csv.writer(file,delimiter=',') # Write the file
		csv_writer.writerow(['***Marker positions and inter-marker differences for autolabeling the "whip forward" frame. Last edit time: ', DTcurr])
		#csv_writer.writerow(f'DataPosN = {DataPosN + 1}')
		csv_writer.writerow(['***Marker indices, names, and positions: mean - sd - N'])
		for iname,name in enumerate(MarkerNames):
			csv_writer.writerow([f'{iname} - {name} - {DataPosM[iname]} - {DataPosS[iname]} - {DataPosN[iname]}'])
		csv_writer.writerow(['***Symmetric matrix of distance means'])
		for irow,row in enumerate(DataDistM):
			csv_writer.writerow(row.astype(int))
		csv_writer.writerow(['***Symmetric matrix of distance SDs'])
		for irow,row in enumerate(DataDistS):
			csv_writer.writerow(row.astype(int))
		csv_writer.writerow(['***Symmetric matrix of distance N'])
		for irow,row in enumerate(DataDistN):
			csv_writer.writerow(row.astype(int))
	print(f'Updated the model with the Position and Inter-marker distance data from the current frame and a few adjacent ones')











def VerifyDist_Get():
	# Try finding a text file in the project folder with the DistMatrix. If found, show
	ProjDir = qtm.settings.directory.get_project_directory()
	file_path = f'{ProjDir}/!TempVerificationThresholds.txt'
	if os.path.exists(file_path):
		print(f'Inspecting {file_path} for more customized thresholds.')
		with open(file_path,'r') as file:
			csv_readobj = csv.reader(file, delimiter=',')
			flag_ReadingMarkNames = False
			MarkerNamesTempCSV = []
			for irow,row in enumerate(csv_readobj):
				if 'Marker Names' in row[0]:
					flag_ReadingMarkNames = True
					continue
				if '***Thresholds***' in row[0]:
					flag_ReadingMarkNames = False
					Dist_Max_Matrix_CSV = np.zeros((len(MarkerNamesTempCSV),len(MarkerNamesTempCSV)),dtype=int)
					break
				if flag_ReadingMarkNames:
					nm = row[0].split(' - ')[1]
					MarkerNamesTempCSV.append(nm)
		Dist_Max_Matrix_CSV = np.genfromtxt(file_path, delimiter=',', dtype=int, skip_header=(irow+1), usecols=np.arange(66))
		
		# List the thresholds
		print('Max Distances:')
		for irow in np.arange(len(Dist_Max_Matrix_CSV)):
			for icol in np.arange(len(Dist_Max_Matrix_CSV)):
				Val = Dist_Max_Matrix_CSV[irow,icol]
				if Val != 0:
					if irow == icol: # Adjacent marker
						print(f'{MarkerNamesTempCSV[irow]}-{MarkerNamesTempCSV[irow+1]}: {Val}')
					else:
						print(f'{MarkerNamesTempCSV[irow]}-{MarkerNamesTempCSV[icol]}: {Val}')
	else:
		print(f'CSV with customized thresholds not found. Run VerifyDist() once to create one, then use _Get() to display them or _Set(e.g. "w1-W_RWristOut",150) to change a value for a pair of markers')
		

def VerifyDist_Set(MarkPair,NewVal):
	ProjDir = qtm.settings.directory.get_project_directory()
	file_path = f'{ProjDir}/!TempVerificationThresholds.txt'
	if os.path.exists(file_path):
		print(f'Inspecting {file_path}.')
		with open(file_path,'r') as file:
			csv_readobj = csv.reader(file, delimiter=',')
			flag_ReadingMarkNames = False
			MarkerNamesTempCSV = []
			for irow,row in enumerate(csv_readobj):
				if 'Marker Names' in row[0]:
					flag_ReadingMarkNames = True
					continue
				if '***Thresholds***' in row[0]:
					flag_ReadingMarkNames = False
					Dist_Max_Matrix_CSV = np.zeros((len(MarkerNamesTempCSV),len(MarkerNamesTempCSV)),dtype=int)
					break
				if flag_ReadingMarkNames:
					nm = row[0].split(' - ')[1]
					MarkerNamesTempCSV.append(nm)
		Dist_Max_Matrix_CSV = np.genfromtxt(file_path, delimiter=',', dtype=int, skip_header=(irow+1), usecols=np.arange(66))
	else:
		print(f'CSV with customized thresholds not found. Run VerifyDist() once to create one, then use _Get() to display them or _Set(e.g. "w1-W_RWristOut",150) to change a value for a pair of markers. Aborting for now.')
		return
	MarkNamesList = MarkPair.split('-')
	if len(MarkNamesList) == 3: # One marker is target as in "Target-2"
		imark1 = MarkNamesList.index('Target')
		mark1 = 'Target' + '-' + MarkNamesList[imark1+1]
		if imark1 > 0:
			mark2 = MarkNamesList[0]
		else:
			mark2 = MarkNamesList[3]
	elif len(MarkNamesList) == 4: # Two markers are target as in "Target-2"
		mark1 = 'Target' + '-' + MarkNamesList[1]
		mark2 = 'Target' + '-' + MarkNamesList[3]
	else:
		mark1 = MarkNamesList[0]
		mark2 = MarkNamesList[1]
	try:
		imark1 = MarkerNamesTempCSV.index(mark1)
		imark2 = MarkerNamesTempCSV.index(mark2)
	except Exception as e:
		print(f'Incorrect marker name specified. The available markers are the following:')
		for i in MarkerNamesTempCSV:
			print(i)
		print('aborting.')
		return
	if imark1 > imark2: # Swap to keep non-zero valuesin the upper-triang matrix
		imark1 = imark1 + imark2
		imark2 = imark1 - imark2
		imark1 = imark1 - imark2
	if imark2 - imark1 == 1:
		OldVal = Dist_Max_Matrix_CSV[imark1,imark1]
		Dist_Max_Matrix_CSV[imark1,imark1] = NewVal
	else:
		OldVal = Dist_Max_Matrix_CSV[imark1,imark2]
		Dist_Max_Matrix_CSV[imark1,imark2] = NewVal
	
	print(f'Threshold for {MarkPair} changed from {OldVal} to {NewVal} mm')
	DTcurr = datetime.now()
	with open(file_path,'w',newline='') as file:
		csv_writer = csv.writer(file,delimiter=',') # Write the file
		csv_writer.writerow(['QTM Verify Dist Thresholds. See the list of included markers, then the matrix of thresholds.  Last edit time: ', DTcurr])
		csv_writer.writerow(['Marker Names and Indices'])
		for iname,name in enumerate(MarkerNamesTempCSV):
			csv_writer.writerow([f'{iname} - {name}'])
		csv_writer.writerow(['***Thresholds***'])
		for irow,row in enumerate(Dist_Max_Matrix_CSV):
			csv_writer.writerow(row)

	




def Verify_GoToNext():
	# Read the temp-file (just created for this recording), get locations of problematic frames,
	# Get the current frame. # Go to the frame right before the closest next issue; Display the range and the markers.

	# TODO: read also the recent DistMatrix, to show not only frame-name-duration, but severity of violation

	ProjDir = qtm.settings.directory.get_project_directory()

	ProjDir = qtm.settings.directory.get_project_directory()
	file_path = f'{ProjDir}/!TempWhipRecordingValidationDistances.npz'
	data = np.load(file_path)
	DistAll = data['var1']
	DistNans = data['var2']
	MarkerNamesReduced = data['var3']


	# Get thresholds
	file_path = f'{ProjDir}/!TempVerificationThresholds.txt'
	if os.path.exists(file_path):
		#print(f'Inspecting {file_path} for more customized thresholds.')
		with open(file_path,'r') as file:
			csv_readobj = csv.reader(file, delimiter=',')
			flag_ReadingMarkNames = False
			MarkerNames = []
			for irow,row in enumerate(csv_readobj):
				if 'Marker Names' in row[0]:
					flag_ReadingMarkNames = True
					continue
				if '***Thresholds***' in row[0]:
					flag_ReadingMarkNames = False
					Dist_Max_Matrix = np.zeros((len(MarkerNames),len(MarkerNames)),dtype=int)
					break
				if flag_ReadingMarkNames:
					nm = row[0].split(' - ')[1]
					MarkerNames.append(nm)
		Dist_Max_Matrix = np.genfromtxt(file_path, delimiter=',', dtype=int, skip_header=(irow+1), usecols=np.arange(66))


	# Get markerNames
	seriesIDs = qtm.data.series._3d.get_series_ids()
	IDs = []
	IDNames = {}
	for id in seriesIDs:
		fullname = qtm.data.object.trajectory.get_label(id)
		if fullname is not None:
			IDs.append(id)
			IDNames[fullname] = id
	NFrames = qtm.gui.timeline.get_frame_count()

	# Process verification report
	file_path = f'{ProjDir}/!TempWhipRecordingValidationReport.txt'
	if os.path.exists(file_path):
		last_mod_t = os.path.getmtime(file_path)
		last_mod_d = datetime.fromtimestamp(last_mod_t)
		cur_t = datetime.now()
		dt_mod = cur_t - last_mod_d
		dt_minutes = dt_mod.total_seconds() / 60
		if dt_minutes > 15:
			print('Last report file was created more than 15 minutes ago. Doublecheck if it is still up to date')
		Prob_Frames_First = []
		Prob_Frames_Dur = []
		Prob_Lines = []
		with open(file_path,'r') as file:
			csv_readobj = csv.reader(file, delimiter=',')
			
			Flag_Reap = 0
			for irow,row in enumerate(csv_readobj):
				if (Flag_Reap==1) & ('***' in row[0]):
					Flag_Reap = 0
				if Flag_Reap:
					if '(' in row[0]:
						#print(row)
						dur = int(row[0].split('(')[1].split(')')[0])
						if dur > 4:
							Prob_Frames_Dur.append(dur)
							Prob_Frames_First.append(int(row[0].split('-')[0]))
							
							#print(row)
							
							if len(row) == 2: # if more than 2 markers on that interval, don't add that information. If less than one, it is 'Handle Markers'
								m1name = row[0].split(': ')[1]
								m2name = row[1].strip(' ')
								#print(m1name + ', ' + m2name)

								# Find the corresponding threshold
								im1 = MarkerNames.index(m1name)
								im2 = MarkerNames.index(m2name)
								if abs(im1-im2) == 1:
									im = min(im1,im2)
									Thresh = Dist_Max_Matrix[im,im]
								else:
									Thresh = max(Dist_Max_Matrix[im1,im2],Dist_Max_Matrix[im2,im1])

								# Find actual distance - not existent, because, due to 32-python in first python-supporting QTM versions, I only saved DistAll for a few special markers.
								# So, do qualisys
								Names = [m1name,m2name]
								MarkerData = getData_forNameList(Names,IDNames,NFrames)
								MarkerData = np.array(MarkerData)
								Dist = np.sqrt(np.sum((MarkerData[:,:,1] - MarkerData[:,:,0]) ** 2,axis=1))
								i0 = int(row[0].split('-')[0])
								i1 = i0 + dur
								Dist_range = Dist[np.arange(i0,i1)]
								Dmax = np.round(np.nanmax(Dist_range),0)
								
								# Find the interval.

								#print(Dist)
								# m1 = MarkerData[:,:,imark1]
								# m2 = MarkerData[:,:,imark2]
								

								rowapp = f'{m1name}-{m2name}: {Dmax} mm exceeded {Thresh} mm'
								Prob_Lines.append(''.join(row).split(': ')[0] + rowapp)
							else:
								Prob_Lines.append(''.join(row))

							# m1name = d12name.split(';')[0]
							# m2name = d12name.split(';')[1]
							# imarkReduced1 = MarkerNamesReduced.index(m1name)
							# imarkReduced2 = MarkerNamesReduced.index(m2name)
							# #if abs(imark2-imark1) == 1: # All markers should be handled fine without using the main diagonal
							# indNan = np.where(DistNans[imarkReduced1,imarkReduced2,:])
							# d12 = DistAll[imarkReduced1,imarkReduced2,:].astype(float)
							# d12[indNan] = np.nan
							# # Display Threshold and peak distance on the interval of interest


				if '*** Overall' in row[0]:
					Flag_Reap = 1
		# for irow,row in enumerate(Prob_Lines):
		# 	print(Prob_Lines[irow])
		# 	print(Prob_Frames_Dur[irow])
		# 	print(Prob_Frames_First[irow])
		
		iCur = qtm.gui.timeline.get_current_frame()
		Prob_Frames_First = np.array(Prob_Frames_First)
		Prob_Frames_First[(Prob_Frames_First-iCur) < 0] = iCur
		idx_next = next((i for i, x in enumerate(Prob_Frames_First - iCur) if x != 0), None)

		# print(Prob_Frames_First - iCur)
		# print(idx_next)
		if idx_next != None:
			#print(Prob_Frames_First[idx_next]-1)
			gtf(Prob_Frames_First[idx_next]-1)
			print(Prob_Lines[idx_next])
		else:
			print('No more problematic ranges >=5 frames which would be after the current frame')
		
	else:
		print('Validation report not found!')











def VerifyDist():
	Whip_CheckWhipM2MDistances()


# Will check some typical issues:
	# whip marker mismatches based on m2m distance
	# right hand marker mismatches based on distances to elbow and to whip handle
	# target 1,2,3 travelling too far

# Some selected marker2marker distances, see this definition completely
def Whip_CheckWhipM2MDistances():
	proj_dir = qtm.settings.directory.get_project_directory()
	print(proj_dir)
	
	seriesIDs = qtm.data.series._3d.get_series_ids()


	# MarkerNames and Max Distance Thresholds

	WhipNames = ['w1','w2','w3','w4','w5','w6','w7','w8','w9','w10','HDR','HDL','HD0']
	Dist_MaxWhip_Seq = [110, 135, 180, 225, 225, 225, 225, 225, 195, 90, 95, 235]
	
	R_ArmNames = ['W_RWristIn','W_RWristOut','W_RHandOut','W_RElbowOut','W_RElbowIn','W_RArm','W_RShoulderTop','W_RShoulderBack']
	Dist_MaxRarm_Seq = [0, 0, 0, 130, 0, 0, 0] # Checking elbows

	L_ArmNames = ['W_LWristIn','W_LWristOut','W_LHandOut','W_LElbowOut','W_LElbowIn','W_LArm','W_LShoulderTop','W_LShoulderBack']
	Dist_MaxLarm_Seq = [0, 0, 0, 120, 0, 0, 0] # Checking elbows

	TorsoNames = ['W_Chest','W_SpineTop','W_BackL','W_BackR','W_WaistLFront','W_WaistLBack','W_WaistRBack','W_WaistRFront']
	Dist_MaxTorso_Seq = [300, 250, 230, 520, 260, 125, 250]  # Changed 3rd value (BackL -- BackR) from 220 to 280 in a large male subject.

	TargetNames = ['Target-1','Target-2','Target-3','Target-R','Target-L']
	Dist_MaxTarg_Seq = [190,235,235,0]

	HeadNames = ['W_HeadTop','W_HeadFront','W_HeadL','W_HeadR']
	Dist_MaxHead_Seq = [150,160,210]

	L_LegNames = ['W_LThigh','W_LKneeOut','W_LKneeIn','W_LShin','W_LAnkleOut','W_LAnkleIn','W_LHeelBack','W_LForefootOut','W_LForefootIn','W_LToeTip']
	Dist_MaxLLeg_Seq = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	R_LegNames = ['W_RThigh','W_RKneeOut','W_RKneeIn','W_RShin','W_RAnkleOut','W_RAnkleIn','W_RHeelBack','W_RForefootOut','W_RForefootIn','W_RToeTip']
	Dist_MaxRLeg_Seq = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	
	

	#print(seriesIDs)
	# get id vs. name
	IDs = []
	IDNames = {}
	for id in seriesIDs:
		fullname = qtm.data.object.trajectory.get_label(id)
		if fullname is not None:
			IDs.append(id)
			IDNames[fullname] = id
	#print(IDNames)
	#print(IDs)
	# get file length in frames
	NFrames = qtm.gui.timeline.get_frame_count()
	
	print(f'\n\n{NFrames} frames in the file. Getting Marker Data...')

	# Get Data from QTM identified trajectories.
	WhipData = getData_forNameList(WhipNames,IDNames,NFrames)
	R_ArmData = getData_forNameList(R_ArmNames,IDNames,NFrames)
	L_ArmData = getData_forNameList(L_ArmNames,IDNames,NFrames)
	TorsoData = getData_forNameList(TorsoNames,IDNames,NFrames)
	TargetData = getData_forNameList(TargetNames,IDNames,NFrames)
	HeadData = getData_forNameList(HeadNames,IDNames,NFrames)
	L_LegData = getData_forNameList(L_LegNames,IDNames,NFrames)
	R_LegData = getData_forNameList(R_LegNames,IDNames,NFrames)
	print('Done')

	# Have these as a matrix saved as a CSV in the project folder. Allow updating it from QTM to not open VSCode, e.g. by calling 
	# UpdateThreshold("w10-W_RHandOut",Value)
	# Upon script initialization, it checks if the csv file is in the folder. If not, it would create one with the values from here. If it is there, it would use values from there.
	# The CSV would have a vertical matrix of number: marker, and then some spacer for parsing, and then a csv-matrix of constraints.


	# Do them jointly
	Names = WhipNames + R_ArmNames + L_ArmNames + TorsoNames + TargetNames + HeadNames + L_LegNames + R_LegNames
	# print(WhipData[0,0,:])
	# print(R_ArmData[0,0,:])
	# print(L_ArmData[0,0,:])
	# print(TorsoData[0,0,:])
	# print(TargetData[0,0,:])
	# print(HeadData[0,0,:])

	MarkData = np.concatenate((WhipData, R_ArmData, L_ArmData, TorsoData, TargetData, HeadData, L_LegData, R_LegData),axis=2)
	DistMaxSeq = Dist_MaxWhip_Seq + [0] + Dist_MaxRarm_Seq + [0] + Dist_MaxLarm_Seq + [0] + Dist_MaxTorso_Seq + [0] + Dist_MaxTarg_Seq + [0] + Dist_MaxHead_Seq + [0] + Dist_MaxLLeg_Seq + [0] + Dist_MaxRLeg_Seq

	# A matrix to specify limits between non-adjacent markers. An old fix.
	DistMatrix = np.full((len(Names),len(Names)),0)
	DistMatrix[37,39] = 400 # Targ-3 to Targ-1

	Relations = ['w10;HDL>w10;HDR','W_RElbowOut;W_RHandOut>W_RElbowOut;W_RWristOut'] # Removed 2nd relation for OR. % For Symmetric Handle (early CL and RH, adding 'W_RHandOut;HDL>W_RHandOut;HDR')
	ValidateByDistances_ShowExceedFrames(Names, MarkData, DistMaxSeq, DistMatrix, Relations, NFrames)

	# FOR EACH OCCURENCE, CHECK IF THERE ARE OTHER SEGMENTS IN THE PROXIMITY (SIMILAR CENTROID, SIMILAR CIRCULAR STD). Auto-fix, report.
	print('Validation finished')
	print('To display current thresholds, run "VerifyDist_Get()". To modify a threshold, run "VeryfyDist_Set(e.g. "w10-W_HeadL", 220)"')



# Written when numpy could not be imported. Later updated, with some old rudiments.
def ValidateByDistances_ShowExceedFrames(MarkerNames, MarkerData, Dist_Max_Seq, Dist_Max_Matrix, Relations, NFrames):
	print('\n Computing inter-marker distance summaries as (mm) Min - Med(IQR perc 5..95) - Max - Number of NonNans; Checking against thresholds.')
	NaN = float('nan')
	# Bring together the sequential check-up and the non-sequential one.
	# A square matrix of size len(MarkerNames). Diagonal elements 0,0 marker 0 with 1, 1,1 marker 1 with 2 etc. with the n,n zero. Off-diagonal elements for the rest.
	#print(np.array(Dist_Max_Seq))
	if len(Dist_Max_Matrix) == 0:
		Dist_Max_Matrix = np.full((len(MarkerNames),len(MarkerNames)),0)
	L = len(Dist_Max_Seq)
	np.fill_diagonal(Dist_Max_Matrix[:L, :L],np.array(Dist_Max_Seq)) # An old approach - for markers in sequence of MarkerNames, have their distance thresholds on the main diagonal.
	#print('Constraints:')
	#print(MarkerNames)
	#print(Dist_Max_Matrix)
	LinesToFile = []
	ReviewFrameFlags = np.zeros((NFrames+1,1),dtype=bool) # In QTM indexing
	ReviewFramesMarkers = {}
	#print(ReviewFrames)
	print(f'Markers No.: {len(MarkerNames)}')

	# for Mbs in range(100,200):
	# 	Len = Mbs * 1024 * 1024 * 8 / 16 / 66 / 66
	# 	LenT = Len / 666
	# 	print(f'{Mbs} Mb -- {np.round(Len,0)} Frames -- {np.round(LenT,2)} s')
	# 	DistAll = np.full((len(MarkerNames),len(MarkerNames),int(np.round(Len,0))), 999, dtype=np.int16)

	### 32-bit Python. Max 128 Mb for a pre-allocated int16 array, corresponding to 66 markers, and 15406 frames and 23 seconds. And 30916 for bool (8-bit why?..)
		# In case NFrames is larger, have more than one array of DistAll and just go through both....
			# But in the current script version, this array is not really being used... Only for Relations and a few special markers. Then, reduce its size to only those markers?...
				# Current solution. redefine the DistAll. And redefine DistNans too
	MarkerNamesReduced = []
	for relexp in Relations:
			d12name = relexp.split('>')[0]
			m1name = d12name.split(';')[0]
			m2name = d12name.split(';')[1]
			d34name = relexp.split('>')[1]
			m3name = d34name.split(';')[0]
			m4name = d34name.split(';')[1]
			MarkerNamesReduced.append(m1name)
			MarkerNamesReduced.append(m2name)
			MarkerNamesReduced.append(m3name)
			MarkerNamesReduced.append(m4name)
	MarkerNamesReduced = np.unique(MarkerNamesReduced).tolist()
	#print(MarkerNamesReduced)
	#return

	# DistAll = np.full((len(MarkerNames),len(MarkerNames),NFrames), 999, dtype=np.int16) # using 999 as sentinel instead of nan, to preserve memory (int, not float), with a Nan-masking bool array.
	# DistNans = np.zeros((len(MarkerNames),len(MarkerNames),NFrames),dtype=bool)

	DistAll = np.full((len(MarkerNamesReduced),len(MarkerNamesReduced),NFrames), 999, dtype=np.int16) # using 999 as sentinel instead of nan, to preserve memory (int, not float), with a Nan-masking bool array.
	DistNans = np.zeros((len(MarkerNamesReduced),len(MarkerNamesReduced),NFrames),dtype=bool)
	print('Inter-marker distance summaries as Min - Med(IQR from 5th to 95th perc.) - Max - N, with frames where distance exceeds a positive threshold, if specified')
	Dist_Max_Matrix = Dist_Max_Matrix.astype(int)
	#print(Dist_Max_Matrix)

	# Try finding a text file in the project folder with the DistMatrix. If found, import from there. If not found, write. 
	ProjDir = qtm.settings.directory.get_project_directory()
	file_path = f'{ProjDir}/!TempVerificationThresholds.txt'
	flag_UpdateCSV = False
	DTcurr = datetime.now()
	if os.path.exists(file_path):
		print(f'Inspecting {file_path} for more customized thresholds.')
		with open(file_path,'r') as file:
			csv_readobj = csv.reader(file, delimiter=',')
			flag_ReadingMarkNames = False
			flag_ReadingDistMatrix = False
			MarkerNamesTempCSV = []
			for irow,row in enumerate(csv_readobj):
				if 'Marker Names' in row[0]:
					flag_ReadingMarkNames = True
					continue
				if '***Thresholds***' in row[0]:
					flag_ReadingMarkNames = False
					Dist_Max_Matrix_CSV = np.zeros((len(MarkerNamesTempCSV),len(MarkerNamesTempCSV)),dtype=int)
					break
				if flag_ReadingMarkNames:
					nm = row[0].split(' - ')[1]
					MarkerNamesTempCSV.append(nm)
		Dist_Max_Matrix_CSV = np.genfromtxt(file_path, delimiter=',', dtype=int, skip_header=(irow+1), usecols=np.arange(66))
			# Compare the two lists of names!
		if MarkerNamesTempCSV != MarkerNames:
			flag_UpdateCSV = True
			print(f'The CSV file was found but the marker list length does not match the prescribed set of markers. Updating the CSV file and using default thresholds from the code')
		if Dist_Max_Matrix_CSV.shape != Dist_Max_Matrix.shape:
			flag_UpdateCSV = True
			print(f'The CSV file was found threshold matrix has unexpected shape. Updating the CSV file and using default thresholds from the code')
	else:
		flag_UpdateCSV = True
	if not(flag_UpdateCSV):
		print(f'Using thresholds from the {file_path}')
		Dist_Max_Matrix = Dist_Max_Matrix_CSV
		#print(Dist_Max_Matrix)
	else:
		print(f'Saving current default threshold values from the code into the file {file_path}.')
		with open(file_path,'w',newline='') as file:
			csv_writer = csv.writer(file,delimiter=',') # Write the file
			csv_writer.writerow(['QTM Verify Dist Thresholds. See the list of included markers, then the matrix of thresholds.  Last edit time: ', DTcurr])
			csv_writer.writerow(['Marker Names and Indices'])
			for iname,name in enumerate(MarkerNames):
				csv_writer.writerow([f'{iname} - {name}'])
			csv_writer.writerow(['***Thresholds***'])
			for irow,row in enumerate(Dist_Max_Matrix):
				csv_writer.writerow(row)



	# Evaluate distances per marker
	skipMarkData = []
	for imark1,m1name in enumerate(MarkerNames):
		if imark1 in skipMarkData:
			continue
		for imark2,m2name in enumerate(MarkerNames):
			if (imark1 == imark2) & (imark1 == (len(MarkerNames)-1)):
				continue
			
			if imark2 in skipMarkData:
				continue
			# Dist = [NaN] * NFrames
			# for iframe,frame in enumerate(MarkerData):
			# 	m1 = frame[imark1]
			# 	m2 = frame[imark2]
			# 	if imark1 == imark2: # for diagonal elements
			# 		m2 = frame[imark2+1]
			# 	if all(not math.isnan(value) for value in m1) and all(not math.isnan(value) for value in m2):  # if using lists and numpy not available
			# 	 	Dist[iframe] = int(round(math.sqrt(sum((x-y) ** 2 for x, y in zip(m1, m2))),0))
			
			# Compute distance for ALL elements, to check relations later.
			if isinstance(MarkerData, list): # Assuming numpy is available.
				MarkerData = np.array(MarkerData)
			m1 = MarkerData[:,:,imark1]
			m2 = MarkerData[:,:,imark2]
			if np.isnan(m1).all():
				print(f'Marker {m1name} does not contain data')
				skipMarkData.append(imark1)
				# DistNans[imark1,:,:] = True
				# DistNans[:,imark1,:] = True
				DistNansTemp = True
				if (m1name in MarkerNamesReduced):
					imarkReduced1 = MarkerNamesReduced.index(m1name)
					DistNans[imarkReduced1,:,:] = True
					DistNans[:,imarkReduced1,:] = True


			if np.isnan(m2).all():
				print(f'Marker {m2name} does not contain data')
				skipMarkData.append(imark2)
				indNan = np.where(np.isnan(Dist))
				# DistNans[imark2,:,:] = True
				# DistNans[:,imark2,:] = True
				if (m2name in MarkerNamesReduced):
					imarkReduced2 = MarkerNamesReduced.index(m2name)
					DistNans[imarkReduced2,:,:] = True
					DistNans[:,imarkReduced2,:] = True

			if imark1 == imark2: # for "diagonal elements"
				m2 = MarkerData[:,:,imark2+1]
				if (imark2+1) in skipMarkData:
					continue
				if np.isnan(m2).all():
					#print(f'Marker {m2name} does not contain data')
					skipMarkData.append(imark2+1)
				
			Dist = np.linalg.norm(m2 - m1,axis=1)
			
			indNan = np.where(np.isnan(Dist))
			DistNansTemp = np.zeros((NFrames,1),dtype=bool)
			DistNansTemp[np.isnan(Dist)]= True

			#DistNans[imark1,imark2,indNan] = True

			indNNan = np.nonzero(~DistNansTemp)[0].tolist()
			
			if (m1name in MarkerNamesReduced) & (m2name in MarkerNamesReduced):
				imarkReduced1 = MarkerNamesReduced.index(m1name)
				imarkReduced2 = MarkerNamesReduced.index(m2name)
				#print(DistNansTemp)
				#print(indNNan)
				DistAll[imarkReduced1,imarkReduced2,indNNan] = Dist[indNNan]
				DistAll[imarkReduced2,imarkReduced1,indNNan] = Dist[indNNan]
				DistNans[imarkReduced1,imarkReduced2,:] = DistNansTemp[:,0]

			# DistAll[imark1,imark2,indNNan] = Dist[indNNan]
			# DistAll[imark2,imark1,indNNan] = Dist[indNNan] # to ensure succesfull addressing later.
			
			if (Dist_Max_Matrix[imark1,imark2] > 0): # Check against constraints, if positive ones are specified. 
				#LinesToFile.append(S)
				#print(S, end='')
				if imark1 == imark2:
					m2name = MarkerNames[imark2+1] # for diagonal elements
				MNamesExp = f'*** {m1name}-{m2name}:'
				Val = Dist_Max_Matrix[imark1,imark2]
				
				Dist_Max_Matrix[imark1,imark2] = 0
				Dist_Max_Matrix[imark2,imark1] = 0 # to avoid repeated computation of the same off-diagonal pair

				# Debug
				#print(f'Markers {MarkerNames[imark1]} and {MarkerNames[imark2]}')
				# Statistical summary of distance.
				Dist_NN = [value for value in Dist if not math.isnan(value)]
				Dist_min = int(min(Dist_NN))
				Dist_max = int(max(Dist_NN))
				Dist_Sort = sorted(Dist_NN)
				length = len(Dist_Sort)
				#Dist_med = int(round((Dist_Sort[length // 2] + Dist_Sort[(length - 1) // 2]) / 2 if length % 2 == 0 else Dist_Sort[length // 2],0)) # if using lists and numpy not available
				Dist_med = int(np.nanmedian(Dist))
				Dist_iqr = int(np.diff(np.nanpercentile(Dist,[5, 95]))) if length >=4 else np.nan
				#q1, q3 = Dist_Sort[(length * 5) // 100], Dist_Sort[(length * 95) // 100] # if using lists and numpy not available
				#Dist_iqr = int(round(q3 - q1, 0)) if length >= 4 else math.nan # if using lists and numpy not available

				# Checkup against the upper-bound constraints, if positive
				if Val > 0:
					Frames_Check = [iframe for iframe, distance in enumerate(Dist) if distance > Val]
					if len(Frames_Check) > 0:
						Frames_Check_Q = [(frame+1) for frame in Frames_Check] # For QTM indexing
						Frames_Check_G = GroupFramesInRanges(Frames_Check_Q)# Group into ranges! 

						#print('\n')
						S = f'{MNamesExp} {Dist_min} - {Dist_med}({Dist_iqr}) - {Dist_max} - n={len(Dist_NN)} *** Exceeded tolerance {Val} at frame(s):'
						LinesToFile.append(S)
						print(S)
						S = Frames_Check_G
						LinesToFile.append(S)
						print(S)

						ReviewFrameFlags[Frames_Check_Q] = True
						for frame in Frames_Check_Q:
							if frame in ReviewFramesMarkers:
								ReviewFramesMarkers[frame] += (f';{m1name};{m2name}')
							else:
								ReviewFramesMarkers[frame] = (f'{m1name};{m2name}')
						#Frames_Check

					elif max(Dist) < 0.8 * Val:
						S = f'{MNamesExp} {Dist_min} - {Dist_med}({Dist_iqr}) - {Dist_max} - n={len(Dist_NN)} *** Within tolerance {Val} OK. Consider decreasing the threshold!'
						LinesToFile.append(S)
						print(S)
					else:
						S = f'{MNamesExp} {Dist_min} - {Dist_med}({Dist_iqr}) - {Dist_max} - n={len(Dist_NN)} *** Within tolerance {Val} OK'
						LinesToFile.append(S)
						print(S)
					# Pack for a final report Frame X: check markers M1, M2....
					


				else:
					S = f'{MNamesExp} {Dist_min} - {Dist_med}({Dist_iqr}) - {Dist_max} - n={len(Dist_NN)} *** No Threshold set'
					LinesToFile.append(S)
					print(S)
				# Prevent repeated (or reverse) checkup
				#Dist_Max_Matrix[imark_1,imark_2] = np.nan
				#Dist_Max_Matrix[imark_2,imark_1] = np.nan
	


	# Check specifically Hd0 so that it is on the line of the whip?
	idxs = [i for i, x in enumerate(MarkerNames) if x in ['w10','HDR','HDL','HD0']]
	Data_Handle = MarkerData[:,:,idxs]
	# from pen-paper solution
	A = Data_Handle[:,:,1]
	B = Data_Handle[:,:,2]
	C = Data_Handle[:,:,0]
	D = Data_Handle[:,:,3]
	e1 = B-A
	e2 = D-C
	n = np.cross(e1,e2,axis=1)
	#print('***')
	#print(n)
	nnorm = np.linalg.norm(n,axis=1)
	#print('***')
	#print(nnorm)
	#conndist = np.dot(n,(A-C).T)# / nnorm# np.dot(n,(A-C),axis=1)/np.dot(n,n,axis=1)
	conndist = abs(np.sum(n * (A-C),axis=1)) / nnorm
	#print('***')
	#print(conndist)

	cmed = int(np.nanmedian(conndist))
	ciqr = int(np.diff(np.nanpercentile(conndist,[25, 75])))
	cmin = int(np.min(conndist[~np.isnan(conndist)]))  
	cmax = int(np.max(conndist[~np.isnan(conndist)]))
	print('***')
	LinesToFile.append('***')
	#print(f'Handle Markers: Distance between vectors (HDR-HDL) and (w10_HD0) is (min-med(iqr)-max) {cmin} - {cmed}({ciqr}) - {cmax}')
	#print('***')
	HdRLTol = 55
	Frames_Check = np.where(conndist > HdRLTol)
	#print(Frames_Check[0].tolist())

	expFormatted = f'Handle Markers: Distance between vectors (HDR-HDL) and (w10_HD0): {cmin} - {cmed}({ciqr}) - {cmax}'

	if len(Frames_Check[0].tolist()) > 0:
		Frames_Check_Q = [(frame+1) for frame in Frames_Check[0].tolist()]
		Frames_Check_G = GroupFramesInRanges(Frames_Check_Q)# Group into ranges! 
		
		S = f'*** {expFormatted} ***Exceeded tolerance {HdRLTol} at frame(s):'
		LinesToFile.append(S)
		print(S)
		S = Frames_Check_G
		LinesToFile.append(S)
		print(S)
		
		ReviewFrameFlags[Frames_Check_Q] = True
		for frame in Frames_Check_Q:
			if frame in ReviewFramesMarkers:
				ReviewFramesMarkers[frame] += (f'; Handle Markers')
			else:
				ReviewFramesMarkers[frame] = (f'Handle Markers')
	else:
		S = f'*** {expFormatted}: *** Within tolerance {HdRLTol} OK'
		LinesToFile.append(S)
		print(S)



	# Check specifically the two head markers
	#print(MarkerNames)
	idxs = [i for i, x in enumerate(MarkerNames) if x in ['W_HeadTop','W_HeadFront','W_HeadL','W_HeadR']]
	#print(idxs)
	Data_Handle = MarkerData[:,:,idxs]
	#print(Data_Handle)
	# from pen-paper solution
	HT = Data_Handle[:,:,0]
	HF = Data_Handle[:,:,1]
	HL = Data_Handle[:,:,2]
	HR = Data_Handle[:,:,3]
	MidHead = 1/2 * (HF + HT)
	#print(MidHead)
	#print(HL)
	#print(HR)
	#np.any(HL[:,0] < MidHead[:,0]) | np.any(HR[:,0] > MidHead[:,0])
	print('***')
	LinesToFile.append('***')
	expFormatted = '*** Head Markers L and R: '
	#print(S,end="")
	#print('***')
	Frames_Check = np.where(((HL[:,0] < MidHead[:,0]) & (~np.isnan(HL[:,0]))) | ((HR[:,0] > MidHead[:,0]) & (~np.isnan(HR[:,0]))))
	if len(Frames_Check[0].tolist()) > 0:
		Frames_Check_Q = [(frame+1) for frame in Frames_Check[0].tolist()]
		Frames_Check_G = GroupFramesInRanges(Frames_Check_Q)# Group into ranges! 
		
		S = f'{expFormatted} *** May be swapped at frame(s):'
		LinesToFile.append(S)
		print(S)
		S = Frames_Check_G
		LinesToFile.append(S)
		print(S)
		
		ReviewFrameFlags[Frames_Check_Q] = True
		for frame in Frames_Check_Q:
			if frame in ReviewFramesMarkers:
				ReviewFramesMarkers[frame] += (f';W_HeadR;W_HeadL')
			else:
				ReviewFramesMarkers[frame] = (f'W_HeadR;W_HeadL')
	else:
		S = f'{expFormatted} *** Correct orientation everywhere, OK'
		LinesToFile.append(S)
		print(S)

	# Save temporarily the most recent DistAll, DistNans, and helpers MarkerNamesReduced
	ProjDir = qtm.settings.directory.get_project_directory()
	file_path = f'{ProjDir}/!TempWhipRecordingValidationDistances.npz'
	np.savez(file_path, var1=DistAll, var2=DistNans, var3=MarkerNamesReduced)

	# Check relational expressions, comparing different distances to each other.
	print('***')
	LinesToFile.append('***')
	S = '*** Relations, reporting Med(IQR 5 to 95) distance for both sides'
	LinesToFile.append(S)
	print(S)

	if len(Relations) > 0:
		#print(Relations)

		for irel,relexp in enumerate(Relations):
			# if irel == 0:
			# 	print('Skipping the w10-HDR < w10-HDL verification')  # Skipped only for symmetric handle (early RH / CL)
			# 	continue
			flag_Skip = False
			d12name = relexp.split('>')[0]
			m1name = d12name.split(';')[0]
			m2name = d12name.split(';')[1]
			imark1 = MarkerNames.index(m1name)
			imark2 = MarkerNames.index(m2name)
			imarkReduced1 = MarkerNamesReduced.index(m1name)
			imarkReduced2 = MarkerNamesReduced.index(m2name)
			#if abs(imark2-imark1) == 1: # All markers should be handled fine without using the main diagonal
			indNan = np.where(DistNans[imarkReduced1,imarkReduced2,:])
			d12 = DistAll[imarkReduced1,imarkReduced2,:].astype(float)
			d12[indNan] = np.nan
			d34name = relexp.split('>')[1]
			m3name = d34name.split(';')[0]
			m4name = d34name.split(';')[1]
			imark3 = MarkerNames.index(m3name)
			imark4 = MarkerNames.index(m4name)
			imarkReduced3 = MarkerNamesReduced.index(m3name)
			imarkReduced4 = MarkerNamesReduced.index(m4name)
			indNan = np.where(DistNans[imarkReduced3,imarkReduced4,:])
			d34 = DistAll[imarkReduced3,imarkReduced4,:].astype(float)
			d34[indNan] = np.nan
			if (d12 == 0).all():
				S = f'Distance {m1name}-{m2name} is all NaN'
				LinesToFile.append(S)
				print(S)
				flag_Skip = True
			if (d34 == 0).all():
				S = f'Distance {m3name}-{m4name} is all NaN'
				LinesToFile.append(S)
				print(S)
				flag_Skip = True
			if flag_Skip: 
				S = f'Skipping the check of this relation'
				LinesToFile.append(S)
				print(S)
				continue

			d12_med = int(np.nanmedian(d12)) if ~np.isnan(np.nanmedian(d12)) else np.nan
			d34_med = int(np.nanmedian(d34)) if ~np.isnan(np.nanmedian(d34)) else np.nan
			d12_iqr = int(np.diff(np.nanpercentile(d12,[5, 95]))) if ~np.isnan(np.diff(np.nanpercentile(d12,[5, 95]))) else np.nan
			d34_iqr = int(np.diff(np.nanpercentile(d34,[5, 95]))) if ~np.isnan(np.diff(np.nanpercentile(d34,[5, 95]))) else np.nan

			expFormatted = f'({m1name}-{m2name}) > ({m3name}-{m4name}) as {d12_med}({d12_iqr}) > {d34_med}({d34_iqr})'

			#print(d12[:20].astype(int))
			#print(d34[:20].astype(int))
			Frames_Check = np.where(d12 < d34)
			#print(Frames_Check[0].tolist())

			if len(Frames_Check[0].tolist()) > 0:
				Frames_Check_Q = [(frame+1) for frame in Frames_Check[0].tolist()]
				Frames_Check_G = GroupFramesInRanges(Frames_Check_Q)# Group into ranges! 
				
				S = f'*** {expFormatted}: *** Violated at frame(s):'
				LinesToFile.append(S)
				print(S)
				S = Frames_Check_G
				LinesToFile.append(S)
				print(S)
				
				ReviewFrameFlags[Frames_Check_Q] = True
				for frame in Frames_Check_Q:
					if frame in ReviewFramesMarkers:
						ReviewFramesMarkers[frame] += (f';{m1name};{m2name};{m3name};{m4name}')
					else:
						ReviewFramesMarkers[frame] = (f'{m1name};{m2name};{m3name};{m4name}')
			else:
				S = f'*** {expFormatted}: *** Satisfied everywhere, OK'
				LinesToFile.append(S)
				print(S)




	# In addition, parse the unidentified trajectories and find the 5 longest ones of those that (fill more than 10 frames)
	IDsAll = qtm.data.series._3d.get_series_ids()
	IDcandFrameLength = {}
	for id in IDsAll:
		if qtm.data.object.trajectory.get_label(id) != None:
			continue
		Data = GetMarkerDataAll(id)
		
		L = len(np.where(~(np.isnan(Data).any(axis=1)))[0])
		#print(f'{id} - {L} - {np.where(~(np.isnan(Data).any(axis=1)))[0]}')
		if L >= 10:
			IDcandFrameLength[id] = L
	#print(IDcandStartTimes)	
	
	#print(IDcandFrameLength)
	#print('_____________')
	IDcandFrameLength = dict(sorted(IDcandFrameLength.items(), key=lambda item: item[1], reverse=True))
	#print(IDcandFrameLength)

	UnindExamineList = []
	IDcandFirstFrame = {}
	IDcandLastFrame = {}
	for iid,id in enumerate(IDcandFrameLength.keys()):
		if iid >=10:
			break
		Data = GetMarkerDataAll(id)
		datarange = np.where(~np.isnan(Data).any(axis=1))[0]
		IDcandFirstFrame[id] = datarange[0] + 1 # for QTM frames.
		IDcandLastFrame[id] = datarange[-1] + 1

	#print(IDcandFirstFrame)

	IDcandFirstFrame = dict(sorted(IDcandFirstFrame.items(), key=lambda item: item[1]))
	for id in IDcandFirstFrame.keys():
		a = IDcandFirstFrame[id]
		b = IDcandLastFrame[id]
		UnindExamineList.append(f'{a}-{b}  --- Traj {id}')
	# Then find the first frame of that trajectory, sort by that first frame, and printout at the very end....
	



	# For a final report, regroup to show "Frame X - check markers A, B, C"
	# Also group them into ranges, if adjacent, and put markers together.
	
	#print(ReviewFramesMarkers)
	if np.any(ReviewFrameFlags):
		ReviewFrames = np.where(ReviewFrameFlags)[0]
		#print(ReviewFrames)
		#return
		Frames_Check_G = []
		Markers_Check_G = []
		for iframe, frame in enumerate(ReviewFrames):

			if iframe == 0: # First frame
				Frame_L = frame
			elif frame - ReviewFrames[iframe-1] > 1: # All other frames for FrameL
				Frame_L = frame

			# if Frame_L == -1:
			# 	print(-1)
			if iframe == (len(ReviewFrames)-1): # Last frame
				#if (Frame_L >= 0): # seems to be always true
				Frame_R = frame	
				if Frame_R > Frame_L:
					# for each appending, find respective markers using ReviewFramesMarkers dictionary and provide them as a concatenated unique string.
					List2Add = ListReviewMarkersInRange(ReviewFramesMarkers,Frame_L,Frame_R)
					FrameLR = f'{Frame_L}-{Frame_R} ({Frame_R-Frame_L+1})'.ljust(15)
					Frames_Check_G.append(f'{FrameLR}: {List2Add}')
				else:
					List2Add = ListReviewMarkersInRange(ReviewFramesMarkers,Frame_L)
					FrameLR = f'{Frame_L}'.ljust(15)
					Frames_Check_G.append(f'{FrameLR}: {List2Add}')
				Frame_R = -1
					
			elif ((ReviewFrames[iframe+1] - frame) > 1): # all other frames for FrameR
				Frame_R = frame
				if Frame_R > Frame_L:
					List2Add = ListReviewMarkersInRange(ReviewFramesMarkers,Frame_L,Frame_R)
					FrameLR = f'{Frame_L}-{Frame_R} ({Frame_R-Frame_L+1})'.ljust(15)
					Frames_Check_G.append(f'{FrameLR}: {List2Add}')
				else:
					List2Add = ListReviewMarkersInRange(ReviewFramesMarkers,Frame_L)
					FrameLR = f'{Frame_L}'.ljust(15)
					Frames_Check_G.append(f'{FrameLR}: {List2Add}')
				Frame_R = -1
		print('***')
		LinesToFile.append('***')
		S = f'*** Overall, review the following markers at these frames: ***'
		LinesToFile.append(S)
		print(S)
		#print(Frames_Check_G)

		for row in Frames_Check_G:
			print(row)
			LinesToFile.append(row)
	else:
		print('***')
		LinesToFile.append('***')
		S = f'*** No problematic frames to review ***'
		LinesToFile.append(S)
		print(S)

				
	print('***')
	LinesToFile.append('***')
	if len(UnindExamineList) > 0:
		S = f'*** There are unidentified or discarded trajectories longer than 10 frames, here are the 10 longest ones: ***'
		LinesToFile.append(S)
		print(S)
		for row in UnindExamineList:
			print(row)
			LinesToFile.append(row)
	else:
		S = f'*** There are no unidentified or discarded trajectories longer than 10 frames. ***'
		LinesToFile.append(S)
		print(S)

	# See if plotting works from QTM...
	a = np.arange(1,10)
	b = a**2
	fig = plt.figure(figsize=(12,3))
	plt.plot(a,b)
	#plt.show()
	fig.savefig(f"{ProjDir}/TESTFIG.png", bbox_inches="tight")

	# Also do it in a Temp text file in the project directory
	ProjDir = qtm.settings.directory.get_project_directory()
	file_path = f'{ProjDir}/!TempWhipRecordingValidationReport.txt'
	DTcurr = datetime.now()
	with open(file_path, 'w') as file:
		file.writelines(DTcurr.strftime("%b %d  %H:%M:%S"))
		file.writelines('\n')
		for row in LinesToFile:
			file.writelines(row)
			file.writelines('\n')
	subprocess.Popen(['start','notepad.exe',file_path], shell=True)








def AddJoinWhip():
	# get all whip markers
	# set current frame around the beginning.
	
	WhipNames = ['w1','w2','w3','w4','w5','w6','w7','w8','w9','w10','HDR','HDL','HD0']
	for iname in WhipNames[::-1]:
		print('***')
		print('***')
		print(f'Auto-adding parts to the marker {iname}')
		print('***')
		print('***')
		time.sleep(1)
		idsel = qtm.data.object.trajectory.find_trajectory(iname)
		print(f'id{idsel}')
		ConnectTrajPartsUsingSupport(idsel)


# A command to use from the console. Calls ConnectTrajPartsUsingSupport, without NpartsAdded limit
def AddJoinTraj(NpartsAddedMax=[]):
	ConnectTrajPartsUsingSupport([],NpartsAddedMax)
	

def AddJoinTrajButton(idsel=[],NpartsAddedMax=[],DebugMode=False):
	ConnectTrajPartsUsingSupport(idsel,NpartsAddedMax,DebugMode)


def ConnectTrajPartsUsingSupport(idsel=[],NpartsAddedMax=[],DebugMode=False):

	# !!! UPD TO-DO
	# At each frame, find trajectories and/or parts that satisfy the relevant constraints.
	# Recompute, for every candidate trajectory/part, find frames which satisfy the constraints.
	# 1. Add those traj/parts which completely satisfy the constraints. Trim overlaps with pre-existing labeled parts.
	# 2. Recompute the remaining ones. Identify overlaps.
	# 3. Add those that do not overlap with the existing ones, starting with the largest ones.
	# 4. Range from longest to shortest. Add one by one
	# Or, when a part is added, always recompute the parts.
	# Alternative approach:
	# Scan candidates by "Typical range in 3D" and "Typical range of distances with 3-4 closest neighbors"
	# Create a boolean table frame vs. candidate. 
	# Mark the longest ones that do not overlap for adding. Trim wrt existing data if necessary.
	# Repeat with shorter ones etc.


	# print(f'Add constraints to prevent HeadL-HeadR swap. And to prevent Hand/WristOut swap') DONE
	# For the head - X is directed leftwards. So HeadR must have X < than HeadTop+HeadFront. 
	# For the wrist/hand - either use a tighter distance constraint or orientation: Wrist-Hand same direction (+-) as ElbowOut-Wrist; 
	# But then if a trajectory jumps within a segment... Then examine segments before adding?..

	# Also design sort of AIM to label everything at the first frame at least?

	# for the current marker, ensure that a few markers closest to it are identified well.
	# using selected marker, identify the closest markers. How?.. Find all centroids, get the three closest ones....
	# Find continuous distances between the current marker and the 3 neighbors. Create constraints lower and upper. 
	# Then get to the next gap, go frame by frame, search among unidentified trajectories in the current frame, see if any satisfy the constraints. 
	# attempt to add them part by part, without splitting.
	Dist_MaxWhip_Seq = [95, 135, 180, 225, 225, 225, 225, 225, 195, 80]
	WhipNames = ['w1','w2','w3','w4','w5','w6','w7','w8','w9','w10','HDR','HDL','HD0']
	R_ArmNames = ['W_RWristIn','W_RWristOut','W_RHandOut','W_RElbowOut','W_RElbowIn','W_RArm','W_RShoulderTop','W_RShoulderBack']
	L_ArmNames = ['W_LWristIn','W_LWristOut','W_LHandOut','W_LElbowOut','W_LElbowIn','W_LArm','W_LShoulderTop','W_LShoulderBack']
	TorsoNames = ['W_Chest','W_SpineTop','W_BackL','W_BackR','W_WaistLFront','W_WaistLBack','W_WaistRBack','W_WaistRFront']
	L_LegNames = ['W_LThigh','W_LKneeOut','W_LKneeIn','W_LShin','W_LAnkleOut','W_LAnkleIn','W_LHeelBack','W_LForefootIn','W_LForefootOut','W_LToeTip']
	R_LegNames = ['W_RThigh','W_RKneeOut','W_RKneeIn','W_RShin','W_RAnkleOut','W_RAnkleIn','W_RHeelBack','W_RForefootIn','W_RForefootOut','W_RToeTip']
	TargetNames = ['Target-1','Target-2','Target-3','Target-R','Target-L']
	HeadNames = ['W_HeadTop','W_HeadFront','W_HeadL','W_HeadR']
	AllTrajNames = WhipNames + R_ArmNames + L_ArmNames + TorsoNames + L_LegNames + R_LegNames + TargetNames + HeadNames
	# Could also apply to whip markers w10..w9 - then do a separate checkup with name detection

	print('***')
	print('***')
	print('***')
	print('***')
	print('***')
	
	NFrames = qtm.gui.timeline.get_frame_count()
	iCur = qtm.gui.timeline.get_current_frame()
	
	#print(NpartsAddedMax)
	#return

	if (not idsel):
		s = qtm.gui.selection.get_selections()
		if s == None:
			print('Nothing is selected')
			return
		if len(s) == 0:
			print('Nothing is selected')
			return
		#print(f'Currently selected: {s}')
		if len(s) > 1:
			print('Too many objects were selected. Aborting..')
			return
		if not ('trajectory' in s[0]['type']):
			print('Something that is not a trajectory was selected. Aborting..')
			return
		idsel = s[0]["id"]
	idname = qtm.data.object.trajectory.get_label(idsel)
	if DebugMode:
		print(f'ID = {idsel}, Name = {name}')
	DataAll = {}
	#DataC = {}
	flag_whip = False
	if idname in WhipNames[:10]:
		flag_whip = True
		AllTrajNames = WhipNames

	for iname,name in enumerate(AllTrajNames):
		#print(f'Getting data from {name}')
		idt = qtm.data.object.trajectory.find_trajectory(name)
		if idt == None:
			continue
		DataAll[name] = GetMarkerDataAll(idt)
		#DataC[name] = np.nanmean(DataAll[name],axis=0)
		if idt == idsel:
			iid = iname
			Dataid = DataAll[name]
	# Get original filling ratio
	Fill_Before = np.count_nonzero(~np.isnan(DataAll[idname]).any(axis=1)) / NFrames * 100

	if np.isnan(Dataid[iCur,1]):
		print('Move the cursor to the left from any gap that you want to be examined and potentially filled. Aborting')
		return
	
	print(f'Currently selected {idname}(ID {idsel})')

	print(NpartsAddedMax)
	if isinstance(NpartsAddedMax, int):
		if NpartsAddedMax == 0:
			NpartsAddedMax = 999999
			print('All possible parts will be added, it might take some extra time, be patient.')
		else:
			print(f'Maximum {NpartsAddedMax} parts will be added, it might take some time, be patient.')
	elif len(NpartsAddedMax) == 0:
		NpartsAddedMax = 30 # Limit maximum number of added parts because long scripts take forever to execute.
		print(f'Maximum {NpartsAddedMax} parts will be added, it might take some time, be patient.')
	else:
		print(f'Maximum {NpartsAddedMax} parts will be added, it might take some time, be patient.')
	#print(NpartsAddedMax)
	#return


	# Get constraints
	NMarkerConstraints = 4 # For the body markers only: the number of closest (median Distance) markers to use as reference distances.
	if flag_whip:		
		# Now, if it is whip, only use two adjacent markers (or one, in case of w1) 
		# Otherwise, use three closest markers.	
		idx = WhipNames.index(idname)
		DataProx, DataProxProx, DataDist, uBProx, uBProxProx, uBDist = AutoAdd_Whip_GetProxDist(DataAll,WhipNames,idx,Dist_MaxWhip_Seq)
	else:
		Dist_Med_Asc, DistIneqIQR, DistIneqMin, DistIneqMax = AutoAdd_Body_ClosestConstraints(DataAll, AllTrajNames, Dataid, idname, NMarkerConstraints)
	# The rest should be the same for Whip and Body, except one condition check later on.


	NpartsAdded = 0
	flag_NpartsContinuePrint = True
	gap_ranges = qtm.data.series._3d.get_gap_ranges(idsel)
	#print(gap_ranges)
	# Only examine gaps which are 3 frames and longer
	gap_ranges_new = []
	for gap in gap_ranges:
		#print(gap)
		if not flag_NpartsContinuePrint:
			continue
		if gap['start'] < iCur:
			continue
		gapL = gap['end'] - gap['start'] + 1
		#gapind = np.arange(gap['start'],gap['end']+1)
		if (len(gap_ranges) > 50) & (gapL <= 3): # Skip short gaps if there're many gaps overall.
			continue
		#print(f'{gap["start"]} and {iCur}, gapL = {gapL}')
		gap_ranges_new.append(gap)
	
	#print(gap_ranges_new)

	# Get all unidentified at the beginning to not repeat that potentially costly step..
	IDsUnind = []
	IDsAll = qtm.data.series._3d.get_series_ids()
	for id in IDsAll:
		if qtm.data.object.trajectory.get_label(id) != None:
			continue
		IDsUnind.append(id)

	print('Getting data of unidentified trajectories...')
	DataUnind = np.full((NFrames,3,len(IDsUnind)),9999,dtype=int)
	DataUnindNanMask = np.full((NFrames,len(IDsUnind)),False,dtype=bool)
	for iidUnind,id in enumerate(IDsUnind):
		Data = GetMarkerDataAll(id)
		fnan = np.isnan(Data).any(axis=1)
		DataUnindNanMask[:,iidUnind] = fnan
		DataUnind[~fnan,:,iidUnind] = Data[~fnan,:].astype(int)


	# DEV. For each candidate, evaluate frames where it satisfies the constraints. I.e. evaluate constraints at every frame
	
	


	print('Looking for such unidentified candidates in each gap that satisfy distance constraints...')
	igap = 0
	TWhileGap = 0
	TWhileMax = 30
	TGap0 = datetime.now()
	while (igap < len(gap_ranges_new)) & (TWhileGap < 30):
	#for igap,gap in enumerate(gap_ranges_new):
		gap = gap_ranges_new[igap]
		if not flag_NpartsContinuePrint:
			igap += 1
			TWhileGap = (datetime.now() - TGap0).total_seconds()
			print(f'Gap time in While {round(TWhileGap,1)} s')
			continue
		gapL = gap['end'] - gap['start'] + 1
		gapind = np.arange(gap['start'],gap['end']+1)
		print(f'Examining gap {igap+1}/{len(gap_ranges_new)}: {gap}, length {gapL} frames')
		flag_candidateTraj_found = False

		# get all trajectories, check unidentified ones, check if data available anywhere within the gap, evaluate constraints. Skip if any exception, continue if a matching candidate identified.
		IDsAll = qtm.data.series._3d.get_series_ids()
		IDcandStartTimes = {}
		for iidUnind,id in enumerate(IDsUnind):

			#Data = GetMarkerDataAll(id)
			Data = np.full((NFrames,3),np.nan)
			fnan = DataUnindNanMask[:,iidUnind]
			Data[~fnan,:] = DataUnind[~fnan,:,iidUnind]

			DataGap = Data[gapind,:]
			if np.isnan(DataGap).all():
				continue
			flag_cont = False
			if flag_whip:
				flag_cont = AutoAdd_Whip_VerifyConstraints(uBProx,uBProxProx,uBDist,gapind,DataGap,DataProx,DataProxProx,DataDist)
			else: 
				flag_cont = AutoAdd_Body_VerifyConstraints(Dist_Med_Asc, gapind, DataAll, DataGap, NMarkerConstraints, DistIneqIQR, DistIneqMin, DistIneqMax, idname)
			if flag_cont:
				continue
			# In case violated not everywhere (<95%), examine by parts. Identify parts
			# If within a part, partial violation, seek to split.


			
			# Once a candidate trajectory found (and not yet rejected), find where it has data and where those data start
			if len(np.where(~np.isnan(Data).any(axis=1))[0]) > 0:
				Temp = np.where(~np.isnan(Data).any(axis=1))[0]
				#print(Temp)
				IDcandStartTimes[id] = Temp[0]
				flag_candidateTraj_found = True
		
		#print(IDcandStartTimes)	
		# Sort them to start from the earliest ones.
		IDcandStartTimes = dict(sorted(IDcandStartTimes.items(), key=lambda item: item[1]))

		
		# For every trajectory candidate, examine parts, try adding all parts, or part by part
		for id in IDcandStartTimes.keys():	
			
			try:
				PartsAll = qtm.data.object.trajectory.get_parts(id)
				PartsAll = [part for part in PartsAll if part['type'] == 'measured']
			except RuntimeError as ert:
				print(f'Could not acquire parts from candidate traj {id}, exception {ert}: continuing to the next candidate')
				continue
			


			flag_partbypart = True
			if len(PartsAll) > 1:
				# Try first adding all at once, to speed things up when parts are too fragmented. If that does not work, go one by one.
				try: 
					qtm.data.object.trajectory.move_parts(id,idsel)
					print(f'Candidate {id}. Succesfully added all {len(PartsAll)} parts to {idname}')
					NpartsAdded = NpartsAdded + 1
					#flag_candidateTraj_found = True
					flag_partbypart = False
				except Exception as e:
					print(f'Could not add all the parts at once, exception: {e}. Will try adding those without overlap')
					# Find the parts which do not go outside of the gap, i.e. no overlaps.
					SubsetParts2Add = []
					for ipart,part in enumerate(PartsAll):
						prange = np.arange(part['range']["start"], (part['range']["end"] + 1)).tolist()
						if np.all(np.isin(prange, gapind)):
							SubsetParts2Add.append(ipart)
					# Try adding them all together
					if len(SubsetParts2Add) > 0:
						try:
							qtm.data.object.trajectory.move_parts(id,idsel,SubsetParts2Add)
							print(f'Candidate {id}. Succesfully added {len(SubsetParts2Add)} parts ({SubsetParts2Add}) to {idname}. Will try to add the rest one-by-one')
							#flag_candidateTraj_found = True
						except Exception as e:
							print(f'Could not add {len(SubsetParts2Add)} non-overlapping parts, exception: {e}. Will go one-by-one now')
					else:
						print('No parts without overlap for this gap. Proceeding to fixswapoverlap')
						

			if flag_partbypart:
				# Re-get the parts which are left
				PartsAll = qtm.data.object.trajectory.get_parts(id)
				PartsAll = [part for part in PartsAll if part['type'] == 'measured']
				#for ipart,part in enumerate(np.flipud(PartsAll)):
				iipart = 0
				ipart = 1
				PartL = len(PartsAll)

				# Simply try adding part by part
				T0 = datetime.now()
				TWhile = 0
				AddedSuccess = []
				while (len(AddedSuccess) < len(PartsAll)) & (TWhile < 2):
				#for ipart,part in enumerate(PartsAll): # Just trying to add them al
				#for ipart,part in enumerate(PartsAll):
					if NpartsAdded >= NpartsAddedMax:
						if flag_NpartsContinuePrint:
							print(f'{NpartsAdded} parts were already added. It is a hard stop for now. Inspect then re-run, if needed')
							DataAll[idname] = GetMarkerDataAll(idsel)
							Fill_After = np.count_nonzero(~np.isnan(DataAll[idname]).any(axis=1)) / NFrames * 100
							if Fill_After > Fill_Before:
								print(f'Fill ratio increased from {np.round(Fill_Before,2)}% to {np.round(Fill_After,2)}%')
							else:
								print(f'No segments added.')
							return
						continue
					#print(f'Candidate {id} - Part {ipart}/{PartL} - {part}. Trying to add to {idname}')
					#print(ipart)
					part = PartsAll[iipart]
					try: 
						qtm.data.object.trajectory.move_parts(id,idsel,[iipart]) # When a part is added, the index is reset in the remaining parts, so part 2 would become part 1.
						print(f'Candidate {id} - Part {ipart}/{PartL} - {part}. Successfully added to {idname}')
						PartsAll = qtm.data.object.trajectory.get_parts(id)
						PartsAll = [part for part in PartsAll if part['type'] == 'measured']
						PartsAll = PartsAll[PartsAll['type'] == 'measured']
						ipart = ipart + 1
						AddedSuccess.append(True)
						NpartsAdded = NpartsAdded + 1
					except Exception as e:
						#print(f'Could not add this part. {e}. Next will examine overlaps')
						print(f'Candidate {id} - Part {ipart}/{PartL} - {part}. Could not add, will explore potential overlap fix')
						AddedSuccess.append(False)
						iipart += 1
						ipart += 1
					
					TWhile = (datetime.now() - T0).total_seconds()
					# print(f'Parts All: {PartsAll}')
					# print(f'Time in While: {TWhile}')
				if TWhile >=2:
					print(f'The While loop timed out (2 s)')




				# Explore overlaps via FIXSWAPOVERLAP
				if ~np.all(AddedSuccess):
					# RE-EVALUATE REMAINING GAPS i.e. one or two within the original gap's range. 
					gap_ranges_temp = qtm.data.series._3d.get_gap_ranges(idsel)
					# Only examine gaps which are 3 frames and longer
					gap_ranges_upd= {}
					for igapp,gapp in enumerate(gap_ranges_temp):
						if gapp['start'] < iCur:
							continue
						gapL = gapp['end'] - gapp['start'] + 1
						if gapL <= 3:
							continue
						#print(f're-evaluated gap {gapp["start"]}-{gapp["end"]}. original gap {gapind[0]}-{gapind[-1]}')
						if (gapp['start'] in gapind) & (gapp['end'] in gapind):
							gapind_upd = np.arange(gapp['start'],gapp['end']+1)
							gap_ranges_upd[igapp] = gapind_upd
					
					Gap_Ranges_Display = []
					# Display them, if any part was added before, therefore changing the gap configuration
					if ~np.all([~item for item in AddedSuccess]):
						for key in gap_ranges_upd.keys():
							Gap_Ranges_Display.append(GroupFramesInRanges(gap_ranges_upd[key].tolist()))
						print(f'Updated gap ranges for attempt: {Gap_Ranges_Display}. original gap {gapind[0]}-{gapind[-1]}')

					# Should be 0, 1, or 2 such gaps.
					for igapp,key in enumerate(gap_ranges_upd.keys()):
						try:
							gapind_upd = gap_ranges_upd[key]
							#print('NOW GOING FOR SWAPPARTSFIXOVERLAP')
							#return
							print('!!!! Using FIXSWAPOVERLAP: distance constraints are not accounted for, check for potential violations !!!!')
							SwapPartsFixOverlap(id,idsel,gapind_upd) # we are using an old gapind which might have changed if some parts were added. This might cause oversplitting.
							#iipart += 1
						except Exception as e:
							print(f'Candidate {id} - could not add part at the range {gapind_upd[0]}-{gapind_upd[-1]}, some error in SwapPartsFixOverlap, exception: {e}')
					#PartsAll = qtm.data.object.trajectory.get_parts(id)	
							
		if not flag_candidateTraj_found:
			print('No suitable candidate trajectories found')
		igap += 1
		TWhileGap = (datetime.now() - TGap0).total_seconds()
		print(f'Gap time in While {round(TWhileGap,1)} s')
	if TWhileGap > TWhileMax:
		print(f'Gap search timed out ({TWhileMax} s)')
	
	DataAll[idname] = GetMarkerDataAll(idsel)
	Fill_After = np.count_nonzero(~np.isnan(DataAll[idname]).any(axis=1)) / NFrames * 100
	if Fill_After > Fill_Before:
		print(f'Fill ratio increased from {np.round(Fill_Before,2)}% to {np.round(Fill_After,2)}%')
	else:
		print(f'No segments added.')

	# If 3-5 markers are selected, they are expected to not move too much with respect to each other, i.e. hand not with torso and whip not with the hand.
	# So in that case, find distance relations among them, and iterate through gaps of each one.




# Helpers for the above
# For Whip and Non-Whip the only difference are conditional checks of candidate trajectories. Isolate them into separate functions.
def AutoAdd_Whip_GetProxDist(DataAll=[],WhipNames=[],idx=[],Dist_MaxWhip_Seq=[]):
	DataProx = DataAll[WhipNames[idx+1]]
	DataProxProx = DataAll[WhipNames[idx+2]]
	uBProx = Dist_MaxWhip_Seq[idx]
	# Experimental
	uBProxProx = []
	if idx < 9:
		uBProxProx = Dist_MaxWhip_Seq[idx] + Dist_MaxWhip_Seq[idx+1]
	#uBProxProx = []

	DataDist = []
	uBDist = []
	#print(f'idx is {idx}')
	if idx > 0:
		DataDist = DataAll[WhipNames[idx-1]]
		uBDist = Dist_MaxWhip_Seq[idx-1]
		print(f'Distance upper bounds: Prox ({WhipNames[idx+1]}) = {uBProx} and Dist ({WhipNames[idx-1]}) = {uBDist}')
	else:
		print(f'Distance upper bounds: Prox ({WhipNames[idx+1]}) = {uBProx} and no Dist')
	# Find the earliest gap. 
	
	return DataProx, DataProxProx, DataDist, uBProx, uBProxProx, uBDist
	
def AutoAdd_Whip_VerifyConstraints(uBProx,uBProxProx,uBDist,gapind,DataGap=[],DataProx=[],DataProxProx=[],DataDist=[]):
	skipFlag = False
	DataGapProx = np.round(np.linalg.norm(DataProx[gapind,:] - DataGap,axis=1))
	DataGapProxProx = np.round(np.linalg.norm(DataProxProx[gapind,:] - DataGap,axis=1))
	if np.nanmedian(DataGapProx) > uBProx:
		skipFlag = True
	if np.array(uBProxProx).size > 0:
		if np.nanmedian(DataGapProxProx) > uBProxProx:
			skipFlag = True
	if len(DataDist) > 0:
		DataGapDist = np.round(np.linalg.norm(DataDist[gapind,:] - DataGap,axis=1))
		if np.nanmedian(DataGapDist) > uBDist:
			skipFlag = True
	return skipFlag


def AutoAdd_Body_ClosestConstraints(DataAll, AllTrajNames, Dataid, idname, NMarkerConstraints):
		# Not a whip.
		Dist_Med = {}
		DistTemp = {}
		DistIneqIQR = {}
		for iname,name in enumerate(AllTrajNames):
			Dist_Med[name] = 9999 # a dummy placeholder instead of nan. Nans are not sorted well.
			if name in DataAll.keys():
				DistTemp[name] = np.round(np.linalg.norm(DataAll[name] - Dataid,axis=1))
				if ~np.isnan(DistTemp[name]).all():
					Dist_Med[name] = int(np.nanmedian(DistTemp[name]))
			#print(f'Dist median {idname}-{name} = {Dist_Med[name]}')
		# Find the N closest ones. parse through them
		Dist_Med_Asc = dict(sorted(Dist_Med.items(), key=lambda item: item[1]))
		print(f'The {NMarkerConstraints} closest markers are')
		DistIneqMin = {}
		DistIneqMax = {}
		for idkey,dkey in enumerate(Dist_Med_Asc.keys()):
			#print(dkey)
			if (idkey == 0) | (idkey >= (NMarkerConstraints+1)):
				continue
			DistIneqMin[dkey], DistIneqMax[dkey] = np.nanpercentile(DistTemp[dkey],[1, 99])
			DistIneqIQR[dkey] = np.diff(np.nanpercentile(DistTemp[dkey],[25,75]))
			print(f'{dkey}, median distance to {idname} = {Dist_Med_Asc[dkey]}. The 1 and 99 perc. bounds are {int(DistIneqMin[dkey])} .. {int(DistIneqMax[dkey])}')
		return Dist_Med_Asc, DistIneqIQR, DistIneqMin, DistIneqMax

def AutoAdd_Body_VerifyConstraints(Dist_Med_Asc, gapind, DataAll, DataGap, NMarkerConstraints, DistIneqIQR, DistIneqMin, DistIneqMax, idname):
	skipFlag = False
	# Verification of constraints for the body (non-whip)
	for idkey, dkey in enumerate(Dist_Med_Asc.keys()):
		if (idkey == 0) | (idkey >= (NMarkerConstraints+1)):
			continue
		Dist = np.round(np.linalg.norm(DataAll[dkey][gapind,:] - DataGap,axis=1))
		# Red flags: a bit weird for now. Perc(Data,5) < 0.9 * Perc(TrainData,1)  or Perc(Data,95) > 1.1 * Perc(TrainData,99)
		#print(f'Candidate ID {id}. Dist to {dkey} must be within {int(DistIneqMin[dkey])}-{int(DistIneqMax[dkey])}. The candidate distances are (5th, med, 95th) {np.nanpercentile(Dist,5)} - {np.nanmedian(Dist)} - {np.nanpercentile(Dist,95)}')
		if (np.nanpercentile(Dist,5) < 0.9*DistIneqMin[dkey]) | (np.nanpercentile(Dist,95) > 1.1*DistIneqMax[dkey]):
			skipFlag = True
			continue
		if (idname == 'W_RWristOut') | (idname == 'W_RHandOut'):
			# Get ElbowOut, WristOut and HandOut. Check if they are in sequence...
			Elb = DataAll['W_RElbowOut']
			Wr = DataAll['W_RWristOut']
			Hd = DataAll['W_RHandOut']
			ElbWrist = np.round(np.linalg.norm(Elb[gapind,:] - Wr[gapind,:],axis=1))
			ElbHd = np.round(np.linalg.norm(Elb[gapind,:] - Hd[gapind,:],axis=1))
			if np.any(ElbWrist > ElbHd):
				skipFlag = True
				continue
		if (idname == 'W_HeadL') | (idname == 'W_HeadR'):
			HL = DataAll['W_HeadL']
			HR = DataAll['W_HeadR']
			HT = DataAll['W_HeadTop']
			HF = DataAll['W_HeadFront']
			Mid = 1/2 * (HT[gapind,:] + HF[gapind,:])
			if np.any(HL[gapind,0] < Mid[:,0]) | np.any(HR[gapind,0] > Mid[:,0]):
				skipFlag = True
				continue
			

	return skipFlag
					
	








def MinDistTimes():
	# select last 3 whip markers, Target-1 marker (average it), HdR and HdL
	# detect: vector mean(HdR,HdL)->mean(last3) is at an angle < ~20deg with the vector mean(HdR,HdL)->Target1mean
	# Take the mean of the matching frames
	# Return a list of i_MinDist
	pass
	


# Wrap around Swap current part W...
# 1. Check if current part has overlaps. If not, also check adjacent parts. Add all which do not overlap with the main one.
# 2. If there is an overlap, trim it and add... Also check adjacent part on the other side (if any)
def SwapPartsFixOverlap(id1=[],id2=[],gaprange=[],SplitOnly = False,DebugMode = False):
#try:
	NFrames = qtm.gui.timeline.get_frame_count()
	#print(f'Total {NFrames} frames')
	iCur = qtm.gui.timeline.get_current_frame()

	# If managing a selection (i.e. not from the AddJoinTraj script), identify the donor and acceptor trajectories
	if (not id1) | (not id2):
		selections = qtm.gui.selection.get_selections()
		print(f'FIXSWAP: Originally selected in this order: {selections}')
		names = []
		ids = []
		if len(selections) > 2:
			print('FIXSWAP: Too many objects were selected. Aborting..')
			return
		for s in selections:
			t = s["type"]
			if t != 'trajectory':
				print('FIXSWAP: At least one of the the selections is not a trajectory. Aborting..')
				return
			ids.append(s["id"])
			names.append(qtm.data.object.trajectory.get_label(s["id"]))
	else:
		ids = [id1, id2]
		names = [qtm.data.object.trajectory.get_label(id1),qtm.data.object.trajectory.get_label(id2)]
	
	if len(ids) < 2:
		print('FIXSWAP: Obly one object selected. Aborting.')
		return

	# Swap to have the donor part the second index.
	if names[0] == None: # swap
		tempID,tempName = ids[0], names[0]
		ids[0], names[0] = ids[1], names[1]
		ids[1], names[1] = tempID,tempName
		#print('Assuming that the second trajectory needs to be added to the first one')
	print(f'FIXSWAP: Will try to add {names[1]}({ids[1]}) or some of its parts to {names[0]}({ids[0]})')
	#print(names)
	#return

	# Get the data and the valid-data flags
	Data0 = GetMarkerDataAll(ids[0])
	Data1 = GetMarkerDataAll(ids[1])
	FlagNNan0 = ~np.isnan(Data0).any(axis=1)
	FlagNNan1 = ~np.isnan(Data1).any(axis=1)
	IndNNan0 = np.where(FlagNNan0)[0]
	IndNNan1 = np.where(FlagNNan1)[0]
	IndNan0 = np.where(~FlagNNan0)[0]

	Parts0All = qtm.data.object.trajectory.get_parts(ids[0])
	Parts1All = qtm.data.object.trajectory.get_parts(ids[1])
	Parts1All = [part for part in Parts1All if part['type'] == 'measured']
	# restrict to onlt the parts in the gaprange if the latter was provided
	# if gap range was not provided, take the one where the cursor is now or, if the cursor is not on a gap, take the next one to the right from the cursor.
	#print(iCur)
	#print(IndNNan0)
	if len(gaprange) == 0:
		if ~FlagNNan0[iCur]: # currently cursor on the gap
			# find edges of that gap
			if min(IndNNan0) >= iCur:
				left_index = 0
			else:
				left_index = max(IndNNan0[((iCur - IndNNan0) > 0)])
			if max(IndNNan0) <= iCur:
				right_index = NFrames
			else:
				right_index = min(IndNNan0[((IndNNan0 - iCur) > 0)])
			gaprange = np.arange(left_index+1, right_index)
		else:
			left_index = min(IndNan0[((IndNan0 - iCur) > 0)])
			right_index = min(IndNNan0[((IndNNan0 - left_index) > 0)])
			gaprange = np.arange(left_index, right_index)
	print(f'FIXSWAP: Examining the gap is {gaprange[0]}-{gaprange[-1]}')
	
	# Only consider parts which overlap with the acceptor part by no more than 1000 frames
	MaxOverlap = 1000
	Parts1All = SelectPartsWithMinimumOverlap(Parts1All,gaprange,MaxOverlap)
	if len(Parts1All) == 0:
		print('FIXSWAP: All parts overlap with existing data larger than 1000 frames. Aborting this gap. Examine manually, other mislabels made earlier may be a cause.')

	# could it be moved as a whole? - when entering from AddJoinTraj, obviously not. But otherwise possibly yes
	if (FlagNNan0 * FlagNNan1 == 0).all():
		try:
			qtm.data.object.trajectory.move_parts(ids[1],ids[0],np.arange(0,len(Parts1All)).tolist())
			print(f'FIXSWAP: All {len(Parts1All)} parts were successfully moved.')
			return
		except Exception as e:
			print(f'FIXSWAP: Could not move all parts at once, exception: {e}')
			print('Trying part by part now')
		ipart = 0
		iipart = 1
		Parts1_L = len(Parts1All)
		T1While = 0
		T00 = datetime.now()
		while (ipart < len(Parts1All)) & (T1While < 2):
			try:
				qtm.data.object.trajectory.move_parts(ids[1],ids[0],[ipart])
				print(f'FIXSWAP: Part ({iipart}/{Parts1_L}) was added succesfully, trying the next one (which has the same index now)')
				iipart = iipart + 1
				Parts1All = qtm.data.object.trajectory.get_parts(ids[1])
				Parts1All = [part for part in Parts1All if part['type'] == 'measured']
				Parts1All = SelectPartsWithMinimumOverlap(Parts1All,gaprange,MaxOverlap)
				if len(Parts1All) == 0:
					print('FIXSWAP: All remaining parts overlap with existing data larger than 1000 frames. Aborting further processing. Examine manually.')
					return
			except Exception as e:
				print(f'FIXSWAP: Part {ipart+1} could not be moved, trying the next one')
				ipart = ipart + 1
			T1While = (datetime.now() - T00).total_seconds()
		if T1While >=2:
			print('FIXSWAP: T1 While timed out (2s)')


	else: # Some parts overlap. Try adding them one by one. If the current part is in question, detect overlap and trim it.
		flag_CurrPart_overlap = False
		ipart = 0
		iipart = 1
		Parts1_L = len(Parts1All)
		T0 = datetime.now()
		TWhile = 0

		while (ipart < len(Parts1All)) & (TWhile < 2):
			curPart = []
			try:
				qtm.data.object.trajectory.move_parts(ids[1],ids[0],[ipart])
				print(f'FIXSWAP: Part ({iipart}/{Parts1_L}) was added succesfully, trying the next one (which has the same index now)')
				iipart = iipart + 1
				#ipart = ipart + 1
				Parts1All = qtm.data.object.trajectory.get_parts(ids[1])
				Parts1All = [part for part in Parts1All if part['type'] == 'measured']
				Parts1All = SelectPartsWithMinimumOverlap(Parts1All,gaprange,MaxOverlap)
				if len(Parts1All) == 0:
					print('FIXSWAP: All remaining parts overlap with existing data larger than 1000 frames. Aborting further processing. Examine manually.')
					return
			except Exception as e:
				print(f'FIXSWAP: Could not add part {iipart}/{Parts1_L}, Exception: {e}')
				SelPart = Parts1All[ipart]
				range_selpart = np.arange(SelPart['range']["start"],SelPart['range']["end"]+1).tolist()
				#print('Current frame part')
				print(f'FIXSWAP: Selected part {iipart}, {SelPart}')
				#print(f'range part {range_selpart[0]} - {range_selpart[-1]}')

				# Find range of the gap. Trim the part to the gap.
				indOverlaps = np.where(FlagNNan0[range_selpart])[0] + range_selpart[0]


				# split at every overlap instance... But wait, here is oversplitting, if a part with up to 1000 overlap is included!
				ipartsplit = 0
				Overlaps2Split = np.unique(np.concatenate((indOverlaps,indOverlaps-1,indOverlaps+1),axis=0))
				# Ensure that splits only occur around the gap edges
				Overlaps2Split = Overlaps2Split[(Overlaps2Split >= gaprange[0]-1) & (Overlaps2Split <= gaprange[-1]+1)]
				#Overlaps2Split = [value for value in Overlaps2Split if value not in range_selpart]
				print(f'FIXSWAP: Overlaps at {GroupFramesInRanges(indOverlaps)}. To split at: {Overlaps2Split}')
				if len(Overlaps2Split) > 10:
					print('FIXSWAP: More than 10 overlaps planned. REVIEW AND DEBUG. Aborting this gap')
					return

				FramesSplitOk = []
				if len(Overlaps2Split) > 1:
					for indO in Overlaps2Split:
						try:
							#print(indO-1)
							qtm.data.object.trajectory.split_part(ids[1],int(indO))
							Parts1All = qtm.data.object.trajectory.get_parts(ids[1])
							Parts1All = [part for part in Parts1All if part['type'] == 'measured']
							Parts1_L = len(Parts1All)
							ipartsplit += 1
							FramesSplitOk.append(indO)
						except Exception as e:
							print(f'FIXSWAP: Could not split the part {ipartsplit + iipart} of traj {ids[1]} at frame {indO-1}, Exception: {e}')
					print(f'FIXSWAP: Splitted trajectory {ids[1]} at frames {FramesSplitOk}')
				

				Parts1All = qtm.data.object.trajectory.get_parts(ids[1])
				Parts1All = [part for part in Parts1All if part['type'] == 'measured']
				Parts1All = SelectPartsWithMinimumOverlap(Parts1All,gaprange,1)
				if len(Parts1All) == 0:
					print('FIXSWAP: The part overlaps with existing data. Aborting further processing. Examine manually.')
					ipart += 1
					continue
				#### SOME ISSUE APPEARS HERE AS IT STOPS PREMATURELY

				print(f'FIXSWAP: Will try adding these parts {Parts1All}')
				Parts1_L = len(Parts1All)
				# Then try adding every part.
				iiipart = 0
				iiiipart = 0
				for part in Parts1All:
					range_selpart = np.arange(part['range']["start"],part['range']["end"]+1).tolist()
					try:
						qtm.data.object.trajectory.move_parts(ids[1],ids[0],[part["indorig"]-iiiipart])
						iiiipart += 1
						print(f'FIXSWAP: Successfully added part ({iiiipart}/{Parts1_L}), range {range_selpart[0] + 1} - {range_selpart[-1] + 1}')
					except Exception as e:
						print(f'FIXSWAP: Could not add part ({iiiipart}/{Parts1_L}), range {range_selpart[0] + 1} - {range_selpart[-1] + 1}, exception: {e}')
						#ipart = ipart + 1
						iiipart = iiipart + 1
						#print('Trying the next one (which has the same index now)')
				ipart = ipart + 1

				# One issue is that the candidate part sometimes conflicts with the other adjacent parts in the acceptor trajectory. QTM would offer to include them in the exchange fund.
				# In case they are short enough, I see it as a seamless swap... Can I 
				#ipart = ipart + 1
			T1 = datetime.now()
			TWhile = (T1 - T0).total_seconds()
		if TWhile >= 2:
			print('FIXSWAP: Part adding While loop timed out (2 s)')
#except Exception as e0:
#	print(f'Some error occurred in SwapPartsFixOverlap: {e0}')


# Helpers
# Only consider parts which overlap with the acceptor part by no more than 1000 frames
def SelectPartsWithMinimumOverlap(PartsAll,gaprange,MaxOverlap):
	PartsUpd = []
	for ipart,part in enumerate(PartsAll):
		if (part['range']["start"] > (gaprange[0] - MaxOverlap)) & (part['range']["end"] < (gaprange[-1] + MaxOverlap)):
			part['indorig'] = ipart
			PartsUpd.append(part)
			#print(f'Will try adding a part {part["range"]["start"]} - {part["range"]["end"]}')
	return PartsUpd
	

		

			
			










##### POSSIBLY OUTDATED AND INOPTIMAL AND NOT WORKING.
		
# To do: make it work when selecting a visible trajectory (possibly identified) and a currently invisible (possibly identified) trajectory via its label.
def SwapPartsFixOverlapOLD(id1=[],id2=[],SplitOnly = False,DebugMode = False):
	# wrapper of "swap current parts (w)"
	# when two markers are selected, this function will
	# detect if they overlap. If no, it would do a simple swap
	# if yes, then at each (1 or two) overlaps, it would select the earlier segment, split it before the overlap, then go to the next frame and swap the segments (w)
	# Get frame; Get both selections; Get the frames where each selection is defined; 
	# Get overlaps. Find location of the overlap wrt the current frame, find length of overlap, find aver(or summary) of distances during overlap(s)
	# 
	NFrames = qtm.gui.timeline.get_frame_count()
	#print(f'Total {NFrames} frames')
	iCur = qtm.gui.timeline.get_current_frame()
	#print(f'Current frame {iCur + 1}')
	if (not id1) | (not id2):
		selections = qtm.gui.selection.get_selections()
		print(f'Currently selected: {selections}')
		names = []
		ids = []
		UnindIncluded = 0
		if len(selections) > 2:
			print('Too many objects were selected. Aborting..')
			return
		for s in selections:
			t = s["type"]
			if t != 'trajectory':
				print('At least one of the the selections is not a trajectory. Aborting..')
				return
			ids.append(s["id"])
			#if qtm.data.object.trajectory.get_label(s["id"]) == None:
			#	UnindIncluded = UnindIncluded + 1
			names.append(qtm.data.object.trajectory.get_label(s["id"]))
		if DebugMode:
			print(names)
		# if UnindIncluded == 0:
		# 	print('One identified and one unidentified trajectory must be selected. Aborting..')
		# 	return
		# elif UnindIncluded == 2:
		# 	print('One identified and one unidentified trajectory must be selected. Aborting..')
		# 	return
		
	else:
		ids = [id1, id2]
		names = [qtm.data.object.trajectory.get_label(id1),qtm.data.object.trajectory.get_label(id2)]


	if np.where(None in names) == 1: # swap
		tempID,tempName = ids[0], names[0]
		ids[0], names[0] = ids[1], names[1]
		ids[1], names[1] = tempID,tempName
	
	# rangeDictMain = qtm.data.series._3d.get_sample_range(ids[0])
	# rangeDictAdd = qtm.data.series._3d.get_sample_range(ids[1])
	# CommonRange = np.arange(min(rangeDictMain['start'],rangeDictAdd['start']),min(rangeDictMain['end'],rangeDictAdd['end'])+1)
	# indMain = np.arange(rangeDictMain['start'],rangeDictMain['end']+1) 
	# indAdd = np.arange(rangeDictAdd['start'],rangeDictAdd['end']+1) 

	DataMain = GetMarkerDataAll(ids[0])
	DataAdd = GetMarkerDataAll(ids[1])
	# Starting from the current frame, find the closest NaN to the left and to the right from each
	RangeMain = FindClosestDataRanges(DataMain,iCur)
	RangeAdd = FindClosestDataRanges(DataAdd,iCur)
	### DEBUG in RaynaH_4_1 at frame 2001 (QTM 2002) w1-None
	# RangeAdd[1,1] = 2018 # - to check OverlapR functionality
	if DebugMode:
		print(RangeMain)
		print(RangeAdd)
	# Keep the longer segment as main
	if (RangeMain[1,1] - RangeMain[1,0]) < (RangeAdd[1,1] - RangeAdd[1,0]):
		print('The unidentified currently visible range is longer than the identified one. Swapping the range arrays.')
		Temp = RangeMain
		RangeMain = RangeAdd
		RangeAdd = Temp
	FramesOverlap = []
	RangeMainAll = np.unique(np.hstack((np.arange(RangeMain[0,0],RangeMain[0,1]+1),np.arange(RangeMain[1,0],RangeMain[1,1]+1),np.arange(RangeMain[2,0],RangeMain[2,1]+1))))
	RangeMainAll = RangeMainAll[~((RangeMainAll < 0) | (RangeMainAll == NFrames ))]
	#print(RangeMainAll)
	for frame in range(RangeAdd[1,0],RangeAdd[1,1]+1):
		#print(frame)
		if frame in RangeMainAll:
			FramesOverlap.append(frame)
	FramesOverlap = np.array(FramesOverlap)


	# UNTIL HERE, IT SHOULD BE CLEARLY DETERMINED WHETHER WE DO MOVE OR SWAP, AND WHICH PART(S) ARE BEING MOVED/SWAPPED
	# i.e. determine current part in both trajectories and the overlap, if any.

	#FramesOverlap = np.array([2001, 2002, 2003, 2007,2008,2009]) # check if that L-R works
	if len(FramesOverlap) > 0:
		print(f'Overlapping frames in 0-index: {FramesOverlap}')
		OverlapL = np.full((1,2),np.nan)
		OverlapR = np.full((1,2),np.nan)
		if ((len(FramesOverlap) == 1) | ((np.diff(FramesOverlap) == 1).all())):
			OverlapL[0,0] = FramesOverlap[0]
			OverlapL[0,1] = FramesOverlap[-1]
		else:
			for iframe,frame in enumerate(FramesOverlap):
				if iframe == 0:
					#pass
					OverlapL[0,0] = frame
				elif frame - FramesOverlap[iframe-1] > 1:
					OverlapL[0,1] = FramesOverlap[iframe-1]
					break
			OverlapR[0,0] = frame
			OverlapR[0,1] = FramesOverlap[-1]
		OverlapL = OverlapL.astype(int)
		if not (np.isnan(OverlapR)).all():
			OverlapR = OverlapR.astype(int)
		if DebugMode:
			print(f'OverlapL: {OverlapL}')
			print(f'OverlapR: {OverlapR}')


		

		# go to the last frame, select TrajAdd, split. Go to +1 frame, 
		#print(f'IDs: {ids}')
		#OverlapL
		#qtm.gui.timeline.set_current_frame(OverlapL[-1])
		#print(OverlapL[0,1])
		#a = OverlapL[0,1]
		PartsRaw = qtm.data.object.trajectory.get_parts(ids[1])
		PartsRaw = [part for part in PartsRaw if part['type'] == 'measured']
		PartRange = np.full((len(PartsRaw),2),np.nan)
		for ipart,part in enumerate(PartsRaw):
			Range = part["range"]
			PartRange[ipart,0] = Range["start"]
			PartRange[ipart,1] = Range["end"]
			if DebugMode:
				print(f'Part {ipart+1} range: {PartRange[ipart,:].astype(int)}')
			if (OverlapL[0,1].item() >= Range["start"]) & (OverlapL[0,1].item() <= Range["end"]):
				curPart = ipart

		
		if DebugMode:
			print('Debugging, no split')
		else:
			try:
				qtm.data.object.trajectory.split_part(ids[1],OverlapL[0,1].item())
				print(f'Splitting Traj-{ids[1]}({names[1]}) after frame {OverlapL[0,1].item()}')
			except	Exception as e:
				print(f'Splitting raised an exception: {e}')

		if not (np.isnan(OverlapR)).all():
			if DebugMode:
				print('Debugging, no split')
			else:
				try:
					qtm.data.object.trajectory.split_part(ids[1],(OverlapR[0,0].item()-1))
					print(f'Splitting Traj-{ids[1]}({names[1]}) after frame {(OverlapR[0,0].item()-1)}')
				except	Exception as e:
					print(f'Splitting raised an exception: {e}')
		
		if not SplitOnly:
			
			if DebugMode:
				print('Debugging, no move')
			else:
				try:
					qtm.data.object.trajectory.move_parts(ids[1],ids[0],[curPart+1])
					print(f'Part {curPart+2} of Traj-{ids[1]}({names[1]}) was moved to Traj-{ids[0]}({names[0]})') # Part + 2 because "Next" part and because indexing in QTM from 1, not from 0.
				except	Exception as e:
					print(f'Moving the part {curPart+2} of Traj-{ids[1]}({names[1]}) to Traj-{ids[0]}({names[0]}) raised an exception: {e}')
			#print('Successfully split and swapped')

		#print('Splitted trajectory')
		#qtm.gui.timeline.set_current_frame(OverlapL[-1]+1)
		#qtm.data.object.trajectory.get_part(ids[1],2)
		#qtm.gui.send_command("split_trajectory")
		
	else:
		if not SplitOnly:
			# Find the current part
			PartsRaw = qtm.data.object.trajectory.get_parts(ids[1])
			PartsRaw = [part for part in PartsRaw if part['type'] == 'measured']
			PartRange = np.full((len(PartsRaw),2),np.nan)
			for ipart,part in enumerate(PartsRaw):
				Range = part["range"]
				PartRange[ipart,0] = Range["start"]
				PartRange[ipart,1] = Range["end"]
				if (iCur < Range["end"]) & (iCur > Range["start"]):
					curPart = ipart
			
			# just swap.
			
			if DebugMode:
				print('Debugging, no move')
			else:
				try:
					qtm.data.object.trajectory.move_parts(ids[1],ids[0],[curPart])
					print(f'Part {curPart+1} of Traj-{ids[1]}({names[1]}) was moved to Traj-{ids[0]}({names[0]}) without splitting.') # Part + 1 because indexing in QTM from 1, not from 0.
				except	Exception as e:
					print(f'Swapping the part {curPart+2} of Traj-{ids[1]}({names[1]}) to Traj-{ids[0]}({names[0]}) raised an exception: {e}')

	# Use the current file RaynaH_4_1 at frame 2002, w1. to develop and debug.










# HELPERS
		
def getData_forNameList(NameList,IDNames,NFrames):
	#NaN = float('nan')
	#DataList = [[[NaN] * 3 for _ in range(len(NameList))] for _ in range(NFrames)] 

	Data = np.full((NFrames,3,len(NameList)),np.nan)
	
	for iw, wname in enumerate(NameList):
		#print(wname)
		if wname in IDNames:
			ID = IDNames[wname]
		else:
			print(f"{wname}''s ID not found")
			continue
		#print(f'')
		#if iw == 0:
		flag_datafound = 0
		try:
			Data[:,:,iw] = GetMarkerDataAll(ID)
		except Exception as e:
			print('Could not fetch the data, exception: {e}')

		# Samp = qtm.data.series._3d.get_samples(ID,{"start":0,"end":NFrames-1})
		# #print(type(Samp))
		# #print(Samp.shape())
		# #print(Samp)
		# #print(f)
		
		# for iframe in range(NFrames-1):
		# 	#print(iframe)
		# 	#print(Samp[iframe]['position'])
		# 	try:
		# 		#a = 1
		# 		#DataList[iframe][iw][0] = int(round(Samp[iframe]['position'][0],0))
		# 		#DataList[iframe][iw][1] = int(round(Samp[iframe]['position'][1],0))
		# 		#DataList[iframe][iw][2] = int(round(Samp[iframe]['position'][2],0))
		# 		Data[iframe,0,iw] = int(round(Samp[iframe]['position'][0],0))
		# 		if not flag_datafound:
		# 			print(f'{iw} - id{ID} - {wname}. Getting data. First valid frame (XYZ, mm): {DataList[iframe][iw]}')
		# 			flag_datafound = True
		# 	except TypeError:
		# 		pass		   
		# 		#print(Samp[iframe]['position'])
		# 	#print(WhipData[:][iw])
		# if not flag_datafound:
		# 	print(f'{iw} - id {ID} - {wname}. Getting data. No valid data! *****')
	#DataArr = np.array(DataList)
	return Data


# Group into ranges for the report in Distance Verification
def GroupFramesInRanges(Frames):						
	Frames_G = []
	if (len(Frames) == 2):
		if abs(np.diff(Frames)) == 1:
			Frames_G.append(f'{Frames[0]}-{Frames[1]} ({2})')
	for iframe, frame in enumerate(Frames):
		CharComma = ''
		if len(Frames_G) > 0:
			CharComma = ', '
		if iframe == 0: # First frame
			Frame_L = frame
		elif frame - Frames[iframe-1] > 1: # All other frames for FrameL
			Frame_L = frame

		if iframe == (len(Frames)-1): # Last frame
			#if (Frame_L >= 0): # seems to be always true
			Frame_R = frame	
			if Frame_R > Frame_L:
				Frames_G.append(f'{CharComma}{Frame_L}-{Frame_R}({Frame_R-Frame_L+1})')
			else:
				Frames_G.append(f'{CharComma}{Frame_L}')
			Frame_R = -1
			Frame_L = -1
				
		elif ((Frames[iframe+1] - frame) > 1): # all other frames for FrameR
			Frame_R = frame
			if Frame_R > Frame_L:
				Frames_G.append(f'{CharComma}{Frame_L}-{Frame_R}({Frame_R-Frame_L+1})')
			else:
				Frames_G.append(f'{CharComma}{Frame_L}')
			Frame_R = -1
			Frame_L = -1
	return Frames_G
			

# Split and format some lines for the report in Distance Verification
def ListReviewMarkersInRange(ReviewMarkers,Frame_L,Frame_R = -1):
	ReviewMarkList = []
	if Frame_R > -1:
		for frame in range(Frame_L,Frame_R+1):
			Marks2Add = ReviewMarkers[frame].split(';')
			for mark in Marks2Add:
				ReviewMarkList.append(mark) 
		#print(f'{Frame_L} - {Frame_R}: {ReviewMarkList}')
	else:
		Marks2Add = (ReviewMarkers[Frame_L].split(';'))
		for mark in Marks2Add:
			ReviewMarkList.append(mark) 
		#print(f'{Frame_L}: {ReviewMarkList}')
	ReviewMarkS = set(ReviewMarkList)
	ReviewMarkList = sorted(list(ReviewMarkS))
	#print(f'{Frame_L} - {Frame_R}: {ReviewMarkList}')
	ReviewMarkStr = ', '.join(ReviewMarkList)
	#print(f'{Frame_L} - {Frame_R}: {ReviewMarkList}')
	return ReviewMarkStr
	#return ReviewMarkList


		
def find_name_by_id(my_dict, target_id):
    for name, id_number in my_dict.items():
        if id_number == target_id:
            return name
    return None  # Return None if no matching ID is found


def find_id_by_name(my_dict, target_name):
    for name, id_number in my_dict.items():
        if name == target_name:
            return id_number
    return None  # Return None if no matching ID is found


def GetMarkerDataAll(id):
	#id = ids[0]
	NFrames = qtm.gui.timeline.get_frame_count()
	Pos = np.full((NFrames,3),np.nan)
	#print(f'NFrames = {NFrames}')
	DataRaw = qtm.data.series._3d.get_samples(id,{"start":0,"end":NFrames-1})
	for iframe,frame in enumerate(DataRaw):
		if frame == None:
			continue
		PosRaw = frame['position']
		Pos[iframe, :] = [int(round(value,0)) for value in PosRaw]
		# print(f'{iframe} --- {Pos[iframe,:]}')
		# if iframe > 5:
		# 	return
	# Pos = int(round(Pos,0))
	return Pos


def FindClosestDataRanges(Data,iCur):
	# Current frame has data.
	# Find the first frame to the left with Nan and then the next frame with Data
	# Similarly the first frame to the right with Nan and then the next frame with Data

	NFrames = qtm.gui.timeline.get_frame_count()
	# print(Data[(iCur-20):(iCur+20)])
	DataRanges = np.full((3,2),np.nan)
	DataRanges[0,:] = - 1
	DataRanges[1,0] = - 1
	DataRanges[1,1] = NFrames
	DataRanges[2,:] = NFrames

	Val = 0
	for i in range(iCur, -1, -1):
		#print(i)
		if Val == 0:
			if np.isnan(Data[i]).any():
				DataRanges[1,0] = i + 1
				Val = 1
				continue
		elif Val == 1:
			if not np.isnan(Data[i]).any():
				DataRanges[0,1] = i
				#leftGap[0] = i # for QTM, add 1
				Val = 2
				continue
		else:
			if np.isnan(Data[i]).any():
				DataRanges[0,0] = i + 1
				break
	Val = 0
	for i in range(iCur,NFrames-1):
		#print(i)
		if Val == 0:
			if np.isnan(Data[i]).any():
				DataRanges[1,1] = i - 1
				Val = 1
				continue
		elif Val == 1:
			if not np.isnan(Data[i]).any():
				DataRanges[2,0] = i
				Val = 2
				continue
		else:
			if np.isnan(Data[i]).any():
				DataRanges[2,1] = i - 1
				break

	return DataRanges.astype(int)







	
# Add a menu
def add_menu():
	menu_id = qtm.gui.insert_menu_submenu(None,"Whip Scripts")
	#_reload_script_modules()
	#custom_menu_bar_instance = Classes.custom_menu_bar_class.custom_menu_bar()

	
	add_command("_GoToFrame", gtf)
	add_command("_Whip_Names_to_W_Names", Whip_Names_to_W_Names)
	add_command("_Whip_W_Names_to_Names", Whip_W_Names_to_Names)
	#add_command("_WhipTrimBits", WhipTrimBits)
	
	
	#add_command("_Whip_CheckWhipM2MDistances", VerifyDist_SetThreshold)
	#add_command("_Whip_CheckWhipM2MDistances", VerifyDist_ShowThresholds)
	add_command("_Whip_CheckWhipM2MDistances", Whip_CheckWhipM2MDistances)
	add_command("_SwapPartsFixOverlap", SwapPartsFixOverlap)
	#add_command("_FindTypicalOrbit", FindTypicalOrbit)

	add_command("_ConnectTrajPartsUsingSupport", AddJoinTrajButton)
	add_command("_MarkerRemoveVirtualParts", MarkerRemoveVirtualParts)
	add_command("_Verify_GoToNext", Verify_GoToNext)
	
	# add_command("markerset_help", lambda:(print(__doc__)))
	#GoToFrame
	add_menu_item(menu_id, "Go to Frame (Terminal)", "_GoToFrame")
	add_menu_item(menu_id, "Whip - add prefix W_ to body markers", "_Whip_Names_to_W_Names")
	add_menu_item(menu_id, "Whip - remove prefix W_ from body markers", "_Whip_W_Names_to_Names")
	#add_menu_item(menu_id, "Whip - trim the first and last 5 frames", "_WhipTrimBits")

	qtm.gui.insert_menu_separator(menu_id)
	add_menu_item(menu_id, "Whip - verify M2M Dist in Whip, RHand, Target (>>VerifyDist)", "_Whip_CheckWhipM2MDistances")
	add_menu_item(menu_id, "Whip - go to the next verify-problematic interval", "_Verify_GoToNext")
	add_menu_item(menu_id, "Swap parts while fixing overlap", "_SwapPartsFixOverlap")
	#add_menu_item(menu_id, "Find typical orbit (subfunction)", "_FindTypicalOrbit")
	add_menu_item(menu_id, "Fill gaps in the marker using the closest markers >>AddJoinTraj()", "_ConnectTrajPartsUsingSupport")
	add_menu_item(menu_id, "Remove virtual parts from the selected marker, if any", "_MarkerRemoveVirtualParts")

	qtm.gui.set_accelerator({"ctrl": True, "alt": False, "shift": True, "key": "w"},"_Whip_Names_to_W_Names")
	qtm.gui.set_accelerator({"ctrl": True, "alt": False, "shift": True, "key": "d"},"_Whip_CheckWhipM2MDistances")
	qtm.gui.set_accelerator({"ctrl": False, "alt": False, "shift": False, "key": "e"},"_SwapPartsFixOverlap")
	qtm.gui.set_accelerator({"ctrl": True, "alt": False, "shift": False, "key": "g"},"_ConnectTrajPartsUsingSupport")
	qtm.gui.set_accelerator({"ctrl": False, "alt": False, "shift": False, "key": "n"},"_Verify_GoToNext")
	
if __name__ == "__main__":
    add_menu()

