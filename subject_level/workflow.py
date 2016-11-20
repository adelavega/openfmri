from glob import glob
import os

from nipype import LooseVersion
from nipype import Workflow, Node, MapNode, JoinNode
from nipype.interfaces import (fsl, Function, ants, nipy)
import nipype.interfaces.freesurfer as fs
from nipype.interfaces.utility import Merge, IdentityInterface
from nipype.utils.filemanip import filename_to_list
from nipype.interfaces.io import DataSink
import nipype.algorithms.modelgen as model
import nipype.algorithms.rapidart as ra
from nipype.algorithms.confounds import TSNR
from nipype.workflows.fmri.fsl import (create_modelfit_workflow,
																			 create_fixed_effects_flow)
import numpy as np

from bids.grabbids import BIDSLayout

### Import workflows and helpers from other files
from registration import create_reg_workflow, create_fs_reg_workflow
from preprocessing_tools import imports, build_filter1, bandpass_filter, extract_noise_components, median, rename

version = 0
if fsl.Info.version() and \
		LooseVersion(fsl.Info.version()) > LooseVersion('5.0.6'):
		version = 507

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

def create_workflow(bids_dir, args, fs_dir, derivatives, workdir, outdir):
		""" Create a workflow for each individual w/o using an infosource """
		layout = BIDSLayout(bids_dir)

		task_id = args.task
		if task_id not in layout.get_tasks():
				raise Exception('task-{} is not found in your dataset'.format(task_id))

		if not os.path.exists(workdir):
				os.makedirs(workdir)
		subjs_to_analyze = []
		if args.subject:
				subjs_to_analyze = ["sub-" + sub_id for sub_id in args.subject]
		else:
				subjs_to_analyze = [sub.split('/')[-1] for sub in layout.get_subjects(return_type='dir')]
		## Maybe have to convert to ints?

		# the master workflow, with subject specific inside
		meta_wf = Workflow('meta_level')

		for subj_label in subjs_to_analyze: #replacing infosource      
				run_id, conds, TR, slice_times = get_subjectinfo(subj_label, bids_dir, 
																												 task_id, args.model)
				# replacing datasource
				bold_files = sorted([f.filename for f in \
											layout.get(subject = subj_label.replace('sub-',''),
											type='bold', task=task_id, extensions=['nii.gz', 'nii'])])

				anat = [f.filename for f in \
								layout.get(subject = subj_label.replace('sub-',''),
								type='T1w', extensions=['nii.gz', 'nii'])][0]

				####
				### Maybe get events here? 
				####
				name = '{sub}_task-{task}'.format(sub=subj_label, task=task_id)

				# until slice timing is fixed, don't use
				kwargs = dict(bold_files=bold_files,
											anat=anat,
											target_file=args.target_file,
											subject_id=subj_label,
											task_id=task_id,
											model_id=args.model,
											TR=TR,
											slice_times=None,
											behav=behav,
											fs_dir=fs_dir,
											conds=conds,
											run_id=run_id,
											highpass_freq=args.hpfilter,
											lowpass_freq=args.lpfilter,
											fwhm=args.fwhm,
											contrast=contrast_file,
											use_derivatives=derivatives,
											outdir=os.path.join(out_dir, subj_label, args.task),
											name=name)
											
				wf = analyze_bids_dataset(**kwargs)
				meta_wf.add_nodes([wf])
		return meta_wf

def get_subjectinfo(subject_id, base_dir, taskname, model, session_id=None):
		"""Get info for a given subject
		Parameters
		----------
		subject_id : string
				Subject identifier (e.g., sub001)
		base_dir : string
				Path to base directory of the dataset
		task : str
				Which task to process (task-%s)
		model_id : int
				Which model to process
		Returns
		-------
		run_ids : list of ints
				Run numbers
		conds : list of str
				Condition names
		TR : float
				Repetition time
		"""
		import json
		model = json.load(open(model, 'r'))

		task = 'task-{}'.format(taskname)
		
		if len(model) == 0:
				raise ValueError('No condition info found in models')
		
		from bids.grabbids import BIDSLayout
		layout = BIDSLayout(base_dir)
		run_ids = layout.get(target='run', return_type='id')
		run_ids = [int(run[4:]) for run in run_ids]

		## Extract conditions from events.tsv, onsets etc....
		import pandas as pd
		pd.read_csv('')

		if session_id:
				json_info = glob(os.path.join(base_dir, subject_id, session_id, 
																			'func','*%s*.json'%(task)))[0]
		else:    
				json_info = glob(os.path.join(base_dir, subject_id, 'func',
																		 '*%s*.json'%(task)))[0]
		if os.path.exists(json_info):
				import json
				with open(json_info, 'rt') as fp:
						data = json.load(fp)
						TR = data['RepetitionTime']
						slice_times = data['SliceTiming']
		else:
				raise Exception("no task info json!")

		### Return a list of conditions, with onsets, duration, etc
		### Convert the whole pipeline to use bunches
		return run_ids[0], conds[n_tasks.index(task)], TR, slice_times
def analyze_bids_dataset(bold_files, 
												 anat, 
												 subject_id, 
												 task_id, 
												 model_id, 
												 TR,
												 behav,
												 slice_times=None,
												 target_file=None, 
												 fs_dir=None, 
												 conds=None,
												 run_id=None,
												 highpass_freq=0.1,
												 lowpass_freq=0.1, 
												 fwhm=6., 
												 contrast=None,
												 use_derivatives=True,
												 num_slices=None,
												 pe_key=None,
												 readout=None,
												 outdir=None, 
												 name='tfmri'):
		
		# Initialize subject workflow and import others
		wf = Workflow(name=name)
		modelfit = create_modelfit_workflow()
		modelfit.inputs.inputspec.interscan_interval = TR
		fixed_fx = create_fixed_effects_flow()

		# Start of bold analysis
		if pe_key:
				infosource = Node(IdentityInterface(fields=['bold', 'pe']),
											name='infosource')
				infosource.iterables = [('bold', bold_files), ('pe', pe_key)]
				infosource.synchronize = True
		else:
				infosource = Node(IdentityInterface(fields=['bold']),
											name='infosource')
				infosource.iterables = ('bold', bold_files)

		"""
		realign each functional run

		Outputs:

		out_file: (a list of items which are an existing file name)
				Realigned files (per run)
		par_file: (a list of items which are an existing file name)
				Motion parameter files. Angles are not euler angles
		"""
		# if no slice_times, SpatialRealign algorithm used
		realign_run = Node(nipy.SpaceTimeRealigner(), 
											 name='realign_per_run')
		if slice_times:
				realign_run.inputs.slice_times = slice_times
				realign_run.inputs.slice_info = 2
				realign_run.inputs.tr = TR
		realign_run.plugin_args = {'sbatch_args': '-c 4'}
		wf.connect(infosource, 'bold', realign_run, 'in_file')

		# Comute TSNR on realigned data regressing polynomials upto order 2
		tsnr = Node(TSNR(regress_poly=2), name='tsnr')
		tsnr.plugin_args = {'qsub_args': '-pe orte 4',
											 'sbatch_args': '--mem=16G -c 4'}

		# Compute the median image across runs (now MapNode)
		calc_median = Node(Function(input_names=['in_files'],
																output_names=['median_file'],
																function=median, imports=imports),
													name='median')
		
		# regardless of topup makes workflow easier to connect
		recalc_median = calc_median.clone(name='recalc_median')


		def run_combiner(bold_file, realign_movpar):
				""" joiner for non-topup """
				return bold_file, realign_movpar

		joiner = JoinNode(Function(input_names=['bold_file',
																						'realign_movpar'],
															 output_names=['corrected_bolds',
																						 'nipy_realign_pars'],
															 function=run_combiner),
											joinsource='infosource',
											joinfield=['bold_file', 'realign_movpar'],
											name='run_joiner')
		wf.connect(realign_run, 'out_file', joiner, 'bold_file')
		wf.connect(realign_run, 'par_file', joiner, 'realign_movpar')

		#realign across runs
		realign_all = realign_run.clone(name='realign_allruns')

		wf.connect(joiner, 'corrected_bolds', realign_all, 'in_file')
		wf.connect(realign_all, 'out_file', tsnr, 'in_file')
		wf.connect(tsnr, 'detrended_file', recalc_median, 'in_files')

		# segment and register
		if fs_dir:
				registration = create_fs_reg_workflow()
				registration.inputs.inputspec.subject_id = subject_id
				registration.inputs.inputspec.subjects_dir = fs_dir
				if target_file:
						registration.inputs.inputspec.target_image = target_file
				else:
						registration.inputs.inputspec.target_image = fsl.Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
		else:
				registration = create_reg_workflow()
				registration.inputs.inputspec.anatomical_image = anat
				registration.inputs.inputspec.target_image = fsl.Info.standard_image('MNI152_T1_2mm.nii.gz')
				registration.inputs.inputspec.target_image_brain = fsl.Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
				registration.inputs.inputspec.config_file = 'T1_2_MNI152_2mm'
		wf.connect(recalc_median, 'median_file', registration, 'inputspec.mean_image')

		""" Quantify TSNR in each freesurfer ROI """
		get_roi_tsnr = MapNode(fs.SegStats(), iterfield=['in_file'], 
													 name='get_aparc_tsnr')
		get_roi_tsnr.inputs.default_color_table = True
		get_roi_tsnr.inputs.avgwf_txt_file = True
		wf.connect(tsnr, 'tsnr_file', get_roi_tsnr, 'in_file')
		wf.connect(registration, 'outputspec.aparc', get_roi_tsnr, 'segmentation_file')

		# Get a brain mask
		mask = Node(fsl.BET(), name='mask-bet')
		mask.inputs.mask = True
		wf.connect(recalc_median, 'median_file', mask, 'in_file')

		""" Detect outliers in a functional imaging series"""
		art = MapNode(ra.ArtifactDetect(),
									iterfield=['realigned_files', 
														 'realignment_parameters'],
									name="art")
		art.inputs.use_differences = [True, False]
		art.inputs.use_norm = True
		art.inputs.norm_threshold = 1
		art.inputs.zintensity_threshold = 3
		art.inputs.mask_type = 'spm_global'
		art.inputs.parameter_source = 'NiPy'
		wf.connect([(realign_all, art, [('out_file', 'realigned_files')]),
								(joiner, art, [('nipy_realign_pars', 'realignment_parameters')])
								])

		def selectindex(files, idx):
				""" Utility function for registration seg files """
				import numpy as np
				from nipype.utils.filemanip import filename_to_list, list_to_filename
				return list_to_filename(np.array(filename_to_list(files))[idx].tolist())

		# could run into problem with mapnode with filesaving
		def motion_regressors(motion_params, order=0, derivatives=1):
				"""Compute motion regressors upto given order and derivative
				motion + d(motion)/dt + d2(motion)/dt2 (linear + quadratic)"""
				out_files = []
				for idx, filename in enumerate(filename_to_list(motion_params)):
						params = np.genfromtxt(filename)
						out_params = params
						for d in range(1, derivatives + 1):
								cparams = np.vstack((np.repeat(params[0, :][None, :], d, axis=0),
																		 params))
								out_params = np.hstack((out_params, np.diff(cparams, d, axis=0)))
						out_params2 = out_params
						for i in range(2, order + 1):
								out_params2 = np.hstack((out_params2, np.power(out_params, i)))
						filename = os.path.join(os.getcwd(), "motion_regressor%02d.txt" % idx)
						np.savetxt(filename, out_params2, fmt="%.10f")
						out_files.append(filename)
				return out_files

		
		motreg = MapNode(Function(input_names=['motion_params', 'order',
																					 'derivatives'],
															output_names=['out_files'],
															function=motion_regressors,
															imports=imports),
										 iterfield=['motion_params'],
										 name='getmotionregress')
		wf.connect(joiner, 'nipy_realign_pars', motreg, 'motion_params')
		
		# Create a filter to remove motion and art confounds
		createfilter1 = MapNode(Function(input_names=['motion_params', 'comp_norm',
																							 'outliers', 'detrend_poly'],
																		 output_names=['out_files'],
																		 function=build_filter1,
																		 imports=imports),
														iterfield=['motion_params'],
														name='makemotionbasedfilter')
		createfilter1.inputs.detrend_poly = 2
		wf.connect(motreg, 'out_files', createfilter1, 'motion_params')
		wf.connect(art, 'norm_files', createfilter1, 'comp_norm')
		wf.connect(art, 'outlier_files', createfilter1, 'outliers')

		filter1 = MapNode(fsl.GLM(out_f_name='F_mcart.nii.gz',
															out_pf_name='pF_mcart.nii.gz',
															demean=True),
											iterfield=['in_file', 'design', 'out_res_name'],
											name='filtermotion')
		wf.connect(realign_all, 'out_file', filter1, 'in_file')
		wf.connect(realign_all, ('out_file', rename, '_filtermotart'), 
							 filter1, 'out_res_name')
		
		# might have problems?
		wf.connect(createfilter1, 'out_files', filter1, 'design')

		createfilter2 = MapNode(Function(input_names=['realigned_file',
																									'mask_file',
																									'num_components',
																									'extra_regressors'],
																		 output_names=['out_files'],
																		 function=extract_noise_components,
																		 imports=imports),
														iterfield=['realigned_file', 'extra_regressors'],
														name='makecompcorrfilter')
		wf.connect(createfilter1, 'out_files', createfilter2, 'extra_regressors')
		wf.connect(filter1, 'out_res', createfilter2, 'realigned_file')
		wf.connect(registration, ('outputspec.segmentation_files', selectindex, [0, 2]),
							 createfilter2, 'mask_file')

		filter2 = MapNode(fsl.GLM(out_f_name='F.nii.gz',
															out_pf_name='pF.nii.gz',
															demean=True),
											iterfield=['in_file', 'design', 'out_res_name'],
											name='filter_noise_nosmooth')
		wf.connect(filter1, 'out_res', filter2, 'in_file')
		wf.connect(filter1, ('out_res', rename, '_cleaned'),
							 filter2, 'out_res_name')
		wf.connect(createfilter2, 'out_files', filter2, 'design')
		wf.connect(mask, 'mask_file', filter2, 'mask')

		bandpass = Node(Function(input_names=['files', 'lowpass_freq',
																					'highpass_freq', 'fs'],
														 output_names=['out_files'],
														 function=bandpass_filter,
														 imports=imports),
										name='bandpass_unsmooth')
		bandpass.inputs.fs = 1. / TR
		bandpass.inputs.highpass_freq = highpass_freq
		bandpass.inputs.lowpass_freq = lowpass_freq
		wf.connect(filter2, 'out_res', bandpass, 'files')

		"""Smooth the functional data using
		:class:`nipype.interfaces.fsl.IsotropicSmooth`.
		"""

		smooth = MapNode(interface=fsl.IsotropicSmooth(), name="smooth", iterfield=["in_file"])
		smooth.inputs.fwhm = fwhm

		wf.connect(bandpass, 'out_files', smooth, 'in_file')

		collector = Node(Merge(2), name='collect_streams')
		wf.connect(smooth, 'out_file', collector, 'in1')
		wf.connect(bandpass, 'out_files', collector, 'in2')

		"""
		Transform the remaining images. First to anatomical and then to target
		"""

		warpall = MapNode(ants.ApplyTransforms(), iterfield=['input_image'],
											name='warpall')
		warpall.inputs.input_image_type = 3
		warpall.inputs.interpolation = 'Linear'
		warpall.inputs.invert_transform_flags = [False, False]
		warpall.inputs.terminal_output = 'file'
		warpall.inputs.reference_image = target_file
		warpall.inputs.args = '--float'
		warpall.inputs.num_threads = 2
		warpall.plugin_args = {'sbatch_args': '-c%d' % 2}

		# transform to target
		wf.connect(collector, 'out', warpall, 'input_image')
		wf.connect(registration, 'outputspec.transforms', warpall, 'transforms')

		mask_target = Node(fsl.ImageMaths(op_string='-bin'), name='target_mask')

		wf.connect(registration, 'outputspec.anat2target', mask_target, 'in_file')

		maskts = MapNode(fsl.ApplyMask(), iterfield=['in_file'], name='ts_masker')
		wf.connect(warpall, 'output_image', maskts, 'in_file')
		wf.connect(mask_target, 'out_file', maskts, 'mask_file')

		# BOLD modeling
		def get_contrasts(contrast_file, task_id, conds):
				""" Setup a basic set of contrasts, a t-test per condition """
				import numpy as np
				import os
				contrast_def = []
				if os.path.exists(contrast_file):
						with open(contrast_file, 'rt') as fp:
								contrast_def.extend([np.array(row.split()) for row in fp.readlines() if row.strip()])
				contrasts = []
				for row in contrast_def:
						if row[0] != 'task-%s' % task_id:
								continue
						con = [row[1], 'T', ['cond%03d' % (i + 1)  for i in range(len(conds))],
									 row[2:].astype(float).tolist()]
						contrasts.append(con)
				# add auto contrasts for each column
				for i, cond in enumerate(conds):
						con = [cond, 'T', ['cond%03d' % (i + 1)], [1]]
						contrasts.append(con)
				return contrasts

		contrastgen = Node(Function(input_names=['contrast_file',
																						 'task_id', 'conds'],
																output_names=['contrasts'],
																function=get_contrasts),
											 name='contrastgen')
		contrastgen.inputs.contrast_file = contrast
		contrastgen.inputs.task_id = task_id
		contrastgen.inputs.conds = conds
		wf.connect(contrastgen, 'contrasts', modelfit, 'inputspec.contrasts')

		def check_behav_list(behav, run_id, conds):
				""" Check and reshape cond00x.txt files """
				import six
				import numpy as np
				num_conds = len(conds)
				if isinstance(behav, six.string_types):
						behav = [behav]
				behav_array = np.array(behav).flatten()
				num_elements = behav_array.shape[0]
				return behav_array.reshape(num_elements/num_conds, num_conds).tolist()

		reshape_behav = Node(Function(input_names=['behav', 'run_id', 'conds'],
																	output_names=['behav'],
																	function=check_behav_list),
												 name='reshape_behav')
		reshape_behav.inputs.behav = behav
		reshape_behav.inputs.run_id = run_id
		reshape_behav.inputs.conds = conds

		modelspec = Node(model.SpecifyModel(),
										 name="modelspec")
		modelspec.inputs.input_units = 'secs'
		modelspec.inputs.time_repetition = TR
		# bold model connections
		wf.connect(reshape_behav, 'behav', modelspec, 'event_files')
		wf.connect(realign_all, 'out_file', modelspec, 'functional_runs')
		wf.connect(joiner, 'nipy_realign_pars', modelspec, 'realignment_parameters')
		# might have problem with mapnode
		wf.connect(art, 'outlier_files', modelspec, 'outlier_files')
		wf.connect(modelspec, 'session_info', modelfit, 'inputspec.session_info')
		wf.connect(realign_all, 'out_file', modelfit, 'inputspec.functional_data')

		def sort_copes(copes, varcopes, contrasts):
				"""Reorder the copes so that now it combines across runs"""
				import numpy as np
				if not isinstance(copes, list):
						copes = [copes]
						varcopes = [varcopes]
				num_copes = len(contrasts)
				n_runs = len(copes)
				all_copes = np.array(copes).flatten()
				all_varcopes = np.array(varcopes).flatten()
				outcopes = all_copes.reshape(len(all_copes)/num_copes, num_copes).T.tolist()
				outvarcopes = all_varcopes.reshape(len(all_varcopes)/num_copes, num_copes).T.tolist()
				return outcopes, outvarcopes, n_runs

		cope_sorter = Node(Function(input_names=['copes', 'varcopes',
																						 'contrasts'],
																output_names=['copes', 'varcopes',
																							'n_runs'],
																function=sort_copes),
											 name='cope_sorter')

		wf.connect(contrastgen, 'contrasts', cope_sorter, 'contrasts')
		wf.connect([(mask, fixed_fx, [('mask_file', 'flameo.mask_file')]),
								(modelfit, cope_sorter, [('outputspec.copes', 'copes')]),
								(modelfit, cope_sorter, [('outputspec.varcopes', 'varcopes')]),
								(cope_sorter, fixed_fx, [('copes', 'inputspec.copes'),
																				 ('varcopes', 'inputspec.varcopes'),
																				 ('n_runs', 'l2model.num_copes')]),
								(modelfit, fixed_fx, [('outputspec.dof_file',
																				'inputspec.dof_files')])])

		def merge_files(copes, varcopes, zstats):
				out_files = []
				splits = []
				out_files.extend(copes)
				splits.append(len(copes))
				out_files.extend(varcopes)
				splits.append(len(varcopes))
				out_files.extend(zstats)
				splits.append(len(zstats))
				return out_files, splits

		mergefunc = Node(Function(input_names=['copes', 'varcopes',
																									'zstats'],
																	 output_names=['out_files', 'splits'],
																	 function=merge_files),
											name='merge_files')
		wf.connect([(fixed_fx.get_node('outputspec'), mergefunc,
																 [('copes', 'copes'),
																	('varcopes', 'varcopes'),
																	('zstats', 'zstats'),
																	])])
		# sure...
		wf.connect(mergefunc, 'out_files', registration, 'inputspec.source_files')

		def split_files(in_files, splits):
				copes = in_files[:splits[0]]
				varcopes = in_files[splits[0]:(splits[0] + splits[1])]
				zstats = in_files[(splits[0] + splits[1]):]
				return copes, varcopes, zstats

		splitfunc = Node(Function(input_names=['in_files', 'splits'],
																		 output_names=['copes', 'varcopes',
																									 'zstats'],
																		 function=split_files),
											name='split_files')
		wf.connect(mergefunc, 'splits', splitfunc, 'splits')
		wf.connect(registration, 'outputspec.transforms',
							 splitfunc, 'in_files')

		if fs_dir:
				get_roi_mean = MapNode(fs.SegStats(default_color_table=True),
																	iterfield=['in_file'], name='get_aparc_means')
				get_roi_mean.inputs.avgwf_txt_file = True
				wf.connect(fixed_fx.get_node('outputspec'), 'copes', get_roi_mean, 'in_file')
				wf.connect(registration, 'outputspec.aparc', get_roi_mean, 'segmentation_file')

				# Sample the average time series in aparc ROIs
				# from rsfmri_vol_surface_preprocessing_nipy.py
				sampleaparc = MapNode(fs.SegStats(default_color_table=True),
														iterfield=['in_file'],
														name='aparc_ts')
				sampleaparc.inputs.segment_id = ([8] + range(10, 14) + [17, 18, 26, 47] +
																			 range(49, 55) + [58] + range(1001, 1036) +
																			 range(2001, 2036))
				sampleaparc.inputs.avgwf_txt_file = True
	
				wf.connect(registration, 'outputspec.aparc', sampleaparc, 'segmentation_file')
				wf.connect(realign_all, 'out_file', sampleaparc, 'in_file')

		"""
		Connect to a datasink
		"""

		def get_subs(subject_id, conds, run_id, task_id):
				subs = [('_subject_id_%s_' % subject_id, '')]
				subs.append(('task_id_%d/' % task_id, '/task%03d_' % task_id))
				subs.append(('bold_dtype_mcf_mask_smooth_mask_gms_tempfilt_mean_warp',
				'mean'))
				subs.append(('bold_dtype_mcf_mask_smooth_mask_gms_tempfilt_mean_flirt',
				'affine'))

				for i in range(len(conds)):
						subs.append(('_flameo%d/cope1.' % i, 'cope%02d.' % (i + 1)))
						subs.append(('_flameo%d/varcope1.' % i, 'varcope%02d.' % (i + 1)))
						subs.append(('_flameo%d/zstat1.' % i, 'zstat%02d.' % (i + 1)))
						subs.append(('_flameo%d/tstat1.' % i, 'tstat%02d.' % (i + 1)))
						subs.append(('_flameo%d/res4d.' % i, 'res4d%02d.' % (i + 1)))
						subs.append(('_warpall%d/cope1_warp.' % i,
												 'cope%02d.' % (i + 1)))
						subs.append(('_warpall%d/varcope1_warp.' % (len(conds) + i),
												 'varcope%02d.' % (i + 1)))
						subs.append(('_warpall%d/zstat1_warp.' % (2 * len(conds) + i),
												 'zstat%02d.' % (i + 1)))
						subs.append(('_warpall%d/cope1_trans.' % i,
												 'cope%02d.' % (i + 1)))
						subs.append(('_warpall%d/varcope1_trans.' % (len(conds) + i),
												 'varcope%02d.' % (i + 1)))
						subs.append(('_warpall%d/zstat1_trans.' % (2 * len(conds) + i),
												 'zstat%02d.' % (i + 1)))
						subs.append(('__get_aparc_means%d/' % i, '/cope%02d_' % (i + 1)))

				for i, run_num in enumerate(run_id):
						subs.append(('__get_aparc_tsnr%d/' % i, '/run%02d_' % run_num))
						subs.append(('__art%d/' % i, '/run%02d_' % run_num))
						subs.append(('__dilatemask%d/' % i, '/run%02d_' % run_num))
						subs.append(('__realign%d/' % i, '/run%02d_' % run_num))
						subs.append(('__modelgen%d/' % i, '/run%02d_' % run_num))
				subs.append(('_bold_dtype_mcf_bet_thresh_dil', '_mask'))
				subs.append(('_output_warped_image', '_anat2target'))
				subs.append(('median_flirt_brain_mask', 'median_brain_mask'))
				subs.append(('median_bbreg_brain_mask', 'median_brain_mask'))
				return subs

		subsgen = Node(Function(input_names=['subject_id', 'conds', 'run_id',
																								 'task_id'],
																	 output_names=['substitutions'],
																	 function=get_subs),
											name='subsgen')
		subsgen.inputs.subject_id = subject_id
		subsgen.inputs.task_id = task_id
		subsgen.inputs.run_id = run_id

		datasink = Node(DataSink(),
										name="datasink")
		datasink.inputs.container = subject_id
		wf.connect(contrastgen, 'contrasts', subsgen, 'conds')
		wf.connect(subsgen, 'substitutions', datasink, 'substitutions')
		wf.connect([(fixed_fx.get_node('outputspec'), datasink,
																 [('res4d', 'res4d'),
																	('copes', 'copes'),
																	('varcopes', 'varcopes'),
																	('zstats', 'zstats'),
																	('tstats', 'tstats')])
																 ])
		wf.connect([(modelfit.get_node('modelgen'), datasink,
																 [('design_cov', 'qa.model'),
																	('design_image', 'qa.model.@matrix_image'),
																	('design_file', 'qa.model.@matrix'),
																 ])])
		wf.connect([(joiner, datasink, [('nipy_realign_pars',
																			'qa.motion')]),
								(mask, datasink, [('mask_file', 'qa.mask')])])
		#wf.connect(registration, 'outputspec.mean2anat_mask', datasink, 'qa.mask.mean2anat')
		wf.connect(art, 'norm_files', datasink, 'qa.art.@norm')
		wf.connect(art, 'intensity_files', datasink, 'qa.art.@intensity')
		wf.connect(art, 'outlier_files', datasink, 'qa.art.@outlier_files')
		wf.connect(registration, 'outputspec.anat2target', datasink, 'qa.anat2target')
		wf.connect(tsnr, 'tsnr_file', datasink, 'qa.tsnr.@map')
		if fs_dir:
				wf.connect(registration, 'outputspec.min_cost_file', datasink, 'qa.mincost')
				wf.connect([(get_roi_tsnr, datasink, [('avgwf_txt_file', 'qa.tsnr'),
																							('summary_file', 'qa.tsnr.@summary')])])
				wf.connect([(get_roi_mean, datasink, [('avgwf_txt_file', 'copes.roi'),
																							('summary_file', 'copes.roi.@summary')])])
				wf.connect(sampleaparc, 'summary_file', datasink, 'timeseries.aparc.@summary')
				wf.connect(sampleaparc, 'avgwf_txt_file', datasink, 'timeseries.aparc')
		wf.connect([(splitfunc, datasink,
								 [('copes', 'copes.mni'),
									('varcopes', 'varcopes.mni'),
									('zstats', 'zstats.mni'),
									])])
		wf.connect(recalc_median, 'median_file', datasink, 'mean')
		wf.connect(registration, 'outputspec.transformed_mean', datasink, 'mean.mni')
		wf.connect(registration, 'outputspec.func2anat_transform', datasink, 'xfm.mean2anat')
		wf.connect(registration, 'outputspec.anat2target_transform', datasink, 'xfm.anat2target')

		"""
		Set processing parameters
		"""

		#modelspec.inputs.high_pass_filter_cutoff = hpcutoff
		modelfit.inputs.inputspec.bases = {'dgamma': {'derivs': use_derivatives}}
		modelfit.inputs.inputspec.model_serial_correlations = True
		modelfit.inputs.inputspec.film_threshold = 1000

		datasink.inputs.base_directory = outdir
		return wf
