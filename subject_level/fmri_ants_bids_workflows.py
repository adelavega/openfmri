#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
=============================================
fMRI: BIDS data, FSL, ANTS, c3daffine
=============================================

A growing number of datasets are available on `OpenfMRI <http://openfmri.org>`_.
This script demonstrates how to use nipype to analyze a BIDS data set::

    python fmri_ants_bids.py /path/to/bids/dir
"""

from nipype import config
import os

from workflow import create_workflow

"""
The following functions run the whole workflow.
"""

if __name__ == '__main__':
    import argparse
    defstr = ' (default %(default)s)'
    parser = argparse.ArgumentParser(prog='fmri_openfmri.py',
                                     description=__doc__)
    parser.add_argument('-d', '--datasetdir', required=True)
    parser.add_argument('-s', '--subject', default=[],
                        nargs='+', type=str,
                        help="Subject name (e.g. 'sub001')")
    parser.add_argument('-m', '--model',
                        help="Model index" + defstr)
    parser.add_argument('-x', '--subjectprefix', default='sub*',
                        help="Subject prefix" + defstr)
    parser.add_argument('-t', '--task', required=True, #nargs='+',
                        type=str, help="Task name" + defstr)
    parser.add_argument('--hpfilter', default=0.1, type=float, 
                        help="High pass frequency (Hz)" + defstr)
    parser.add_argument('--lpfilter', default=0.1, type=float,
                        help="Low pass frequency (Hz)" + defstr)
    parser.add_argument('--fwhm', default=6.,
                        type=float, help="Spatial FWHM" + defstr)
    parser.add_argument('--derivatives', action="store_true",
                        help="Use derivatives" + defstr)
    parser.add_argument("-o", "--output_dir", dest="outdir",
                        help="Output directory base")
    parser.add_argument("-w", "--work_dir", dest="work_dir",
                        help="Output directory base")
    parser.add_argument("-p", "--plugin", dest="plugin",
                        default='Linear',
                        help="Plugin to use")
    parser.add_argument("--plugin_args", dest="plugin_args",
                        help="Plugin arguments")
    parser.add_argument("--sd", dest="fs_dir", default=None,
                        help="FreeSurfer subjects directory (if available)")
    parser.add_argument("--target", dest="target_file",
                        help=("Target in MNI space. Best to use the MindBoggle "
                              "template - only used with FreeSurfer"
                              "OASIS-30_Atropos_template_in_MNI152_2mm.nii.gz"))
    parser.add_argument("--session_id", dest="session_id", default=None,
                        help="Session id, ex. 'ses-1'")
    parser.add_argument("--crashdump_dir", dest="crashdump_dir",
                        help="Crashdump dir", default=None)
    parser.add_argument('--debug', action="store_true",
                        help="Activate nipype debug mode" + defstr)
    args = parser.parse_args()
    data_dir = os.path.abspath(args.datasetdir)
    out_dir = args.outdir
    work_dir = os.getcwd()

    if args.work_dir:
        work_dir = os.path.abspath(args.work_dir)
    if args.outdir:
        outdir = os.path.abspath(args.outdir)
    else:
        outdir = os.path.join(work_dir, 'output')
    if args.crashdump_dir:
        crashdump = os.path.abspath(args.crashdump_dir)
    else:
        crashdump = os.getcwd()

    derivatives = args.derivatives
    if derivatives is None:
       derivatives = False

    fs_dir = args.fs_dir
    if fs_dir is not None:
        fs_dir = os.path.abspath(fs_dir)

    if args.debug:
        from nipype import logging
        config.enable_debug_mode()
        logging.update_logging(config)
        
    wf = create_workflow(data_dir, args, 
                         fs_dir, derivatives,
                         work_dir, out_dir)
    wf.base_dir = work_dir
    
    if args.crashdump_dir:
        wf.config['execution']['crashdump_dir'] = args.crashdump_dir

    if args.plugin_args:
        wf.run(args.plugin, plugin_args=eval(args.plugin_args))
    else:
        wf.run(args.plugin)
