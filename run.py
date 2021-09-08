import os, sys, glob
import shutil
import numpy as np
import pandas as pd
import json
from gromacs.fileformats.mdp import MDP
from dpdispatcher import Machine, Resources, Task, Submission
from analysis import get_energy, calc_diff_free_energy_block_avg, plot_corr
import pandas as pd
import warnings


def make_dir(dirname):
    cnt = len(list(glob.glob(f"{dirname}.bk.*")))
    if os.path.exists(dirname):
        shutil.move(dirname, f"{dirname}.bk.{cnt:03}")
    os.mkdir(dirname)


PREPARE_MD    = 0
RUN_MD        = 1
PREPARE_RERUN = 2
RUN_RERUN     = 3
ANALYZE       = 4

class AutoCorrTask(object):
    def __init__(self, jdata, mdata):
        self.jdata = jdata
        self.mdata = mdata

        self.solvated_md_prefix     = self.jdata.get("solvated_md_prefix", "solvated_md")
        self.solvated_deepmd_prefix = self.jdata.get("solvated_deepmd_prefix", "solvated_deepmd")
        self.complex_md_prefix      = self.jdata.get("complex_md_prefix", "complex_md")
        self.complex_deepmd_prefix  = self.jdata.get("complex_deepmd_prefix", "complex_deepmd")
        self.prefix                 = os.path.abspath(self.jdata["system_prefix"])
        self.task_path              = os.path.abspath(self.jdata["task_path"])

        if isinstance(self.jdata["systems"], list):
            self.systems = self.jdata["systems"]
        elif isinstance(self.jdata["systems"], str):
            self.systems = list(np.loadtxt(self.jdata["systems"], dtype=str))
        else:
            raise RuntimeError("Invalid systems: {}".format(self.jdata["systems"]))

        # make mdp
        self.make_mdp()
        self.sim_time = self.mdp["dt"] * self.mdp["nsteps"]
        # init command
        self.init_command()

        # dpdispatcher
        self.machine     = Machine.load_from_dict(self.mdata["machine"])
        self.resources   = Resources.load_from_dict(self.mdata["resources"])
        self.run_tasks   = []
        self.rerun_tasks = []

        # reuse
        self.reuse_solvated_md = self.jdata.get("reuse_solvated_md", "")
        self.reuse_complex_md = self.jdata.get("reuse_complex_md", "")
        self.reuse_solvated_deepmd = self.jdata.get("reuse_solvated_deepmd", "")
        self.reuse_complex_deepmd = self.jdata.get("reuse_complex_deepmd", "")

        # recording
        self.record_file = "record"
        if not os.path.exists(self.record_file):
            f = open(self.record_file, "w")
            f.close()
            self.status = -1
        else:
            rec = np.loadtxt(self.record_file, dtype=int).flatten()
            self.status = int(rec[-1])
        
        # analysis
        self.analysis  = self.jdata.get("analysis", {"num_block": 5, "unit": "kcal"})
        self.num_block = self.analysis.get("num_block", 5)
        self.unit      = self.analysis.get("unit", "kcal")
        self.start     = self.analysis.get("start", 0)
        self.end       = self.analysis.get("end", self.mdp["nsteps"] // self.mdp["nstenergy"] + 1)
        self.res       = {}
        self.save_corr = self.analysis.get("save_corr", "corr.csv")
        self.exclusions = self.analysis.get("exclusions", [])
        self.add_zero  = self.analysis.get("add_zero", False)
        if not os.path.isabs(self.save_corr):
            self.save_corr = os.path.join(self.task_path, self.save_corr)
        
        self.ori_data = self.analysis.get("ori_data", "results.csv")
        if not os.path.isabs(self.ori_data):
            self.ori_data = os.path.join(self.prefix, self.ori_data)
        if not os.path.isfile(self.ori_data):
            self.ori_data = None
            warnings.warn(f"Original data file not exists: {self.ori_data}")
        else:
            self.ori_data = pd.read_csv(self.ori_data)
            self.ori_data.loc[:, "lig_1"] = [str(l) for l in self.ori_data["lig_1"]]
            self.ori_data.loc[:, "lig_2"] = [str(l) for l in self.ori_data["lig_2"]]
        
        self.outpng = self.analysis.get("out_png", "results.png")
        if not os.path.isabs(self.outpng):
            self.outpng = os.path.join(self.task_path, self.outpng)
        
        self.save_results = self.analysis.get("save_results", "results.csv")
        if not os.path.isabs(self.save_results):
            self.save_results = os.path.join(self.task_path, self.save_results)
        
        sim_time = self.analysis.get("sim_time", None)
        if sim_time is not None:
            assert isinstance(sim_time, int) or isinstance(sim_time, float)
            self.sim_time = sim_time


    def make_mdp(self):
        mdp = MDP(self.jdata["mdp_filename"])
        for setting in self.jdata["mdp_settings"]:
            mdp[setting] = self.jdata["mdp_settings"][setting]
        self.mdp = mdp
        return mdp
    
    def init_command(self):
        group_name = self.jdata.get("group_name", "MOL")
        gmx_cmd    = self.mdata.get("gmx_command", "gmx")

        md_cmd = f"{gmx_cmd} grompp -c npt.gro -f md.mdp -p processed.top -o md.tpr"
        md_cmd += f" && {gmx_cmd} mdrun -deffnm md -cpi"
        md_cmd += f' && echo -e "Potential\\n" | {gmx_cmd} energy -f md.edr -o md_ener.xvg'
        md_cmd += f' && echo -e "{group_name}\\n{group_name}" | {gmx_cmd} trjconv -f md.trr -s em.tpr -o md_traj.gro -pbc mol -center -ur compact'

        deepmd_cmd = "export GMX_DEEPMD_INPUT_JSON=input.json"
        deepmd_cmd += f" && {gmx_cmd} grompp -c npt.gro -f md.mdp -p processed.top -o deepmd.tpr"
        deepmd_cmd += f" && {gmx_cmd} mdrun -deffnm deepmd -cpi"
        deepmd_cmd += f' && echo -e "Potential\\n" | {gmx_cmd} energy -f deepmd.edr -o deepmd_ener.xvg'
        deepmd_cmd += f' && echo -e "{group_name}\\n{group_name}" | {gmx_cmd} trjconv -f deepmd.trr -s em.tpr -o deepmd_traj.gro -pbc mol -center -ur compact'

        self.md_cmd = md_cmd
        self.deepmd_cmd = deepmd_cmd

        md_rerun_cmd = f"{gmx_cmd} mdrun -s md.tpr -rerun deepmd.trr -e md_rerun.edr -g md_rerun.log"
        md_rerun_cmd += f' && echo -e "Potential\\n" | {gmx_cmd} energy -f md_rerun.edr -o md_rerun_ener.xvg'
        md_rerun_cmd += " && rm -rf traj.*"
        md_rerun_cmd += r" && rm -rf \#*"

        deepmd_rerun_cmd = "export GMX_DEEPMD_INPUT_JSON=input.json"
        deepmd_rerun_cmd += f" && {gmx_cmd} mdrun -s deepmd.tpr -rerun md.trr -e deepmd_rerun.edr -g deepmd_rerun.log"
        deepmd_rerun_cmd += f' && echo -e "Potential\n" | {gmx_cmd} energy -f deepmd_rerun.edr -o deepmd_rerun_ener.xvg'
        deepmd_rerun_cmd += " && rm -rf traj.*"
        deepmd_rerun_cmd += r" && rm -rf \#*"

        self.md_rerun_cmd = md_rerun_cmd
        self.deepmd_rerun_cmd = deepmd_rerun_cmd

    def prepare_md(self):
        print("Preparing md...")
        # common settings for md
        for pp in [self.solvated_md_prefix, self.complex_md_prefix, self.solvated_deepmd_prefix, self.complex_deepmd_prefix]:
            make_dir(os.path.join(self.task_path, pp))
            for ss in self.systems:
                make_dir(os.path.join(self.task_path, pp, ss))
                shutil.copyfile(os.path.join(self.prefix, pp, ss, "npt.gro"), os.path.join(self.task_path, pp, ss, "npt.gro"))
                shutil.copyfile(os.path.join(self.prefix, pp, ss, "processed.top"), os.path.join(self.task_path, pp, ss, "processed.top"))
                shutil.copyfile(os.path.join(self.prefix, pp, ss, "em.tpr"), os.path.join(self.task_path, pp, ss, "em.tpr"))
                self.mdp.write(os.path.join(self.task_path, pp, ss, "md.mdp"))
        
        # deepmd required files
        for pp in [self.solvated_deepmd_prefix, self.complex_deepmd_prefix]:
            for ss in self.systems:
                with open(os.path.join(self.task_path, pp, ss, "input.json"), "w") as fp:
                    json.dump(self.jdata["input_json"], fp)
                shutil.copyfile(os.path.join(self.prefix, pp, ss, "type.raw"), os.path.join(self.task_path, pp, ss, "type.raw"))
                shutil.copyfile(os.path.join(self.prefix, pp, ss, "index.raw"), os.path.join(self.task_path, pp, ss, "index.raw"))
            
        if self.reuse_solvated_md:
            for ss in self.systems:
                os.symlink(os.path.join(self.reuse_solvated_md, ss, "md.trr"),
                           os.path.join(self.task_path, self.solvated_md_prefix, ss, "md.trr"))
                os.symlink(os.path.join(self.reuse_solvated_md, ss, "md_ener.xvg"),
                           os.path.join(self.task_path, self.solvated_md_prefix, ss, "md_ener.xvg"))
        
        if self.reuse_complex_md:
            for ss in self.systems:
                os.symlink(os.path.join(self.reuse_complex_md, ss, "md.trr"),
                           os.path.join(self.task_path, self.complex_md_prefix, ss, "md.trr"))
                os.symlink(os.path.join(self.reuse_complex_md, ss, "md_ener.xvg"),
                           os.path.join(self.task_path, self.complex_md_prefix, ss, "md_ener.xvg"))
        
        if self.reuse_solvated_deepmd:
            for ss in self.systems:
                os.symlink(os.path.join(self.reuse_solvated_md, ss, "deepmd.trr"),
                           os.path.join(self.task_path, self.solvated_deepmd_prefix, ss, "deepmd.trr"))
                os.symlink(os.path.join(self.reuse_solvated_md, ss, "deepmd_ener.xvg"),
                           os.path.join(self.task_path, self.solvated_deepmd_prefix, ss, "deepmd_ener.xvg"))
        
        if self.reuse_complex_deepmd:
            for ss in self.systems:
                os.symlink(os.path.join(self.reuse_complex_deepmd, ss, "deepmd.trr"),
                           os.path.join(self.task_path, self.complex_md_prefix, ss, "deepmd.trr"))
                os.symlink(os.path.join(self.reuse_complex_deepmd, ss, "deepmd_ener.xvg"),
                           os.path.join(self.task_path, self.complex_md_prefix, ss, "deepmd_ener.xvg"))
        
        if self.reuse_solvated_md and self.reuse_solvated_deepmd:
            for ss in self.systems:
                os.symlink(os.path.join(self.reuse_solvated_md, ss, "md_rerun_ener.xvg"),
                           os.path.join(self.task_path, self.solvated_md_prefix, ss, "md_rerun_ener.xvg"))
                os.symlink(os.path.join(self.reuse_solvated_deepmd, ss, "deepmd_rerun_ener.xvg"),
                           os.path.join(self.task_path, self.solvated_deepmd_prefix, ss, "deepmd_rerun_ener.xvg"))
        
        if self.reuse_complex_md and self.reuse_complex_deepmd:
            for ss in self.systems:
                os.symlink(os.path.join(self.reuse_complex_md, ss, "md_rerun_ener.xvg"),
                           os.path.join(self.task_path, self.complex_md_prefix, ss, "md_rerun_ener.xvg"))
                os.symlink(os.path.join(self.reuse_complex_deepmd, ss, "deepmd_rerun_ener.xvg"),
                           os.path.join(self.task_path, self.complex_deepmd_prefix, ss, "deepmd_rerun_ener.xvg"))
        
        with open(self.record_file, 'a') as f:
            f.write(str(PREPARE_MD) + "\n")

    
    def run_md(self):
        print("Running md...")
        # solvated md
        if not self.reuse_solvated_md:
            for ss in self.systems:
                task = Task(command=self.md_cmd, task_work_path=os.path.join(self.task_path, self.solvated_md_prefix, ss))
                self.run_tasks.append(task)
        
        # complex md
        if not self.reuse_complex_md:
            for ss in self.systems:
                task = Task(command=self.md_cmd, task_work_path=os.path.join(self.task_path, self.complex_md_prefix, ss))
                self.run_tasks.append(task)
        
        # solvated deepmd
        if not self.reuse_solvated_deepmd:
            for ss in self.systems:
                task = Task(command=self.deepmd_cmd, task_work_path=os.path.join(self.task_path, self.solvated_deepmd_prefix, ss))
                self.run_tasks.append(task)
        
        # complex deepmd
        if not self.reuse_complex_deepmd:
            for ss in self.systems:
                task = Task(command=self.deepmd_cmd, task_work_path=os.path.join(self.task_path, self.complex_deepmd_prefix, ss))
                self.run_tasks.append(task)

        submission = Submission(
            machine   = self.machine,
            resources = self.resources,
            task_list = self.run_tasks,
            work_base = self.task_path, 
        )
        submission.run_submission()

        with open(self.record_file, 'a') as f:
            f.write(str(RUN_MD) + "\n")
    
    def prepare_rerun(self):
        print("Preparing rerun...")
        for ss in self.systems:
            os.symlink(os.path.join(self.task_path, self.solvated_md_prefix,     ss, "md.trr"),
                       os.path.join(self.task_path, self.solvated_deepmd_prefix, ss, "md.trr"))
            os.symlink(os.path.join(self.task_path, self.complex_md_prefix,      ss, "md.trr"),
                       os.path.join(self.task_path, self.complex_deepmd_prefix,  ss, "md.trr"))
            os.symlink(os.path.join(self.task_path, self.solvated_deepmd_prefix, ss, "deepmd.trr"),
                       os.path.join(self.task_path, self.solvated_md_prefix,     ss, "deepmd.trr"))
            os.symlink(os.path.join(self.task_path, self.complex_deepmd_prefix,  ss, "deepmd.trr"),
                       os.path.join(self.task_path, self.complex_md_prefix,      ss, "deepmd.trr"))
        
        with open(self.record_file, 'a') as f:
            f.write(str(PREPARE_RERUN) + "\n")
    
    def run_rerun(self):
        print("Running rerun...")
        for pp in [self.solvated_md_prefix, self.complex_md_prefix]:
            for ss in self.systems:
                task = Task(command=self.md_rerun_cmd, task_work_path=os.path.join(self.task_path, pp, ss))
                self.run_tasks.append(task)
        
        for pp in [self.solvated_deepmd_prefix, self.complex_deepmd_prefix]:
            for ss in self.systems:
                task = Task(command=self.deepmd_rerun_cmd, task_work_path=os.path.join(self.task_path, pp, ss))
                self.run_tasks.append(task)

        submission = Submission(
            machine   = self.machine,
            resources = self.resources,
            task_list = self.run_tasks,
            work_base = self.task_path, 
        )
        submission.run_submission()

        with open(self.record_file, 'a') as f:
            f.write(str(RUN_RERUN) + "\n")
    
    def analyze(self):
        print("Analyzing...")
        for ss in self.systems:
            print(ss, end=" ")
            # solvated 
            solvated_md_dir     = os.path.join(self.task_path, self.solvated_md_prefix)
            solvated_deepmd_dir = os.path.join(self.task_path, self.solvated_deepmd_prefix)
            solvated_eners = get_energy(ss, solvated_md_dir, solvated_deepmd_dir)
            solvated_eners = tuple([e[self.start: self.end].copy() for e in solvated_eners])
            # solvated_diff, solvated_std = calc_diff_free_energy_block_avg(solvated_eners, self.num_block, self.unit)
            solvated_diff, solvated_std = calc_diff_free_energy_block_avg(solvated_eners, 
                                                                          self.num_block,
                                                                          self.unit, 
                                                                          add_zero=self.add_zero)

            # complex
            complex_md_dir     = os.path.join(self.task_path, self.complex_md_prefix)
            complex_deepmd_dir = os.path.join(self.task_path, self.complex_deepmd_prefix)
            complex_eners = get_energy(ss, complex_md_dir, complex_deepmd_dir)
            complex_eners = tuple([e[self.start: self.end].copy() for e in complex_eners])
            # complex_diff, complex_std = calc_diff_free_energy_block_avg(complex_eners, self.num_block, self.unit)
            complex_diff, complex_std = calc_diff_free_energy_block_avg(complex_eners,
                                                                        self.num_block, 
                                                                        self.unit,
                                                                        add_zero=self.add_zero)

            self.res[ss] = {
                "solvated_diff": solvated_diff,
                "solvated_std": solvated_std,
                "complex_diff": complex_diff,
                "complex_std": complex_std
            }
            print("finished")

        corrs     = []
        corr_stds = []
        if self.ori_data is not None:
            for ii in range(self.ori_data.shape[0]):
                lig_1   = str(self.ori_data.loc[ii, "lig_1"])
                lig_2   = str(self.ori_data.loc[ii, "lig_2"])
                exp     = float(self.ori_data.loc[ii, "exp"])
                ori     = float(self.ori_data.loc[ii, "fep"])
                ori_std = float(self.ori_data.loc[ii, "std"])
                
                corr = ori + (self.res[lig_2]["complex_diff"] - self.res[lig_1]["complex_diff"]) - (self.res[lig_2]["solvated_diff"] - self.res[lig_1]["solvated_diff"])
                corr_std = np.linalg.norm([ori_std,
                                           self.res[lig_2]["complex_std"],
                                           self.res[lig_2]["solvated_std"],
                                           self.res[lig_1]["complex_std"],
                                           self.res[lig_1]["solvated_std"]])
                corrs.append(corr)
                corr_stds.append(corr_std)
            self.ori_data["corr"] = corrs
            self.ori_data["corr_std"] = corr_stds
            self.ori_data.to_csv(self.save_results, index=None, float_format="%4f")

            msk = [True for _ in range(self.ori_data.shape[0])]
            for ii in range(self.ori_data.shape[0]):
                if self.ori_data["lig_1"][ii] in self.exclusions or self.ori_data["lig_2"][ii] in self.exclusions:
                    msk[ii] = False
            self.ori_data = self.ori_data[msk]
            
            plot_corr(self.ori_data["fep"],
                      self.ori_data["corr"],
                      self.ori_data["std"],
                      self.ori_data["corr_std"],
                      self.ori_data["exp"],
                      self.outpng,
                      f"{self.sim_time / 1000:.1f}ns",
                      self.unit)

        self.res = pd.DataFrame(self.res)
        self.res.T.to_csv(self.save_corr)

        with open(self.record_file, 'a') as f:
            f.write(str(ANALYZE) + "\n")

    def run(self):
        workflow = [self.prepare_md, self.run_md, self.prepare_rerun, self.run_rerun, self.analyze]
        for work in workflow[self.status + 1:]:
            work()


if __name__ == "__main__":
    param_json   = sys.argv[1]
    machine_json = sys.argv[2]
    with open(param_json, 'r') as f:
        jdata = json.load(f)
    with open(machine_json, 'r') as f:
        mdata = json.load(f)
    
    task = AutoCorrTask(jdata, mdata)
    task.run()

