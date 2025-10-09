#!/usr/bin/env python3
"""
Simple OpenFOAM case runner for AAA geometries.

This script runs pre-generated OpenFOAM cases sequentially or in parallel.
Each case is already complete with:
  - Geometry files (constant/triSurface/*.stl)
  - Mesh dictionaries (system/blockMeshDict, system/snappyHexMeshDict)
  - Boundary conditions (0/U, 0/p)
  - Solver settings (system/controlDict, system/fvSchemes, system/fvSolution)
"""

import os
import subprocess
from pathlib import Path
import shutil
import json


class OpenFOAMCaseRunner:
    """Runner for pre-generated OpenFOAM cases."""

    def __init__(
        self,
        openfoam_bashrc="/opt/spack/v0.23.1/opt/spack/linux-ubuntu24.04-sapphirerapids/gcc-13.3.0/openfoam-org-9-yjq7t3b3xh75m5s7u5bntedww5u7sekn/etc/bashrc",
        solver="pimpleFoam",
        parallel=False,
        num_procs=1,
        run_blockmesh=True,
        run_snappyhexmesh=True,
        run_solver=True
    ):
        """
        Initialize OpenFOAM case runner.

        Args:
            openfoam_bashrc: Path to OpenFOAM bashrc for sourcing environment
            solver: OpenFOAM solver to use (default: pimpleFoam for transient flow)
            parallel: Whether to run solver in parallel with MPI
            num_procs: Number of processors for parallel runs
            run_blockmesh: Whether to run blockMesh
            run_snappyhexmesh: Whether to run snappyHexMesh
            run_solver: Whether to run the CFD solver
        """
        self.openfoam_bashrc = Path(openfoam_bashrc)
        self.solver = solver
        self.parallel = parallel
        self.num_procs = num_procs
        self.run_blockmesh = run_blockmesh
        self.run_snappyhexmesh = run_snappyhexmesh
        self.run_solver = run_solver

        # Verify OpenFOAM installation
        if not self.openfoam_bashrc.exists():
            print(f"⚠️  Warning: OpenFOAM bashrc not found at {self.openfoam_bashrc}")
            print(f"   Attempting to find OpenFOAM installation...")
            self._find_openfoam()

    def _find_openfoam(self):
        """Try to find OpenFOAM installation."""
        common_paths = [
            "/opt/openfoam9/etc/bashrc",
            "/opt/openfoam/etc/bashrc",
            "/usr/lib/openfoam/openfoam9/etc/bashrc",
            Path.home() / "OpenFOAM" / "OpenFOAM-9" / "etc" / "bashrc"
        ]

        for path in common_paths:
            if Path(path).exists():
                self.openfoam_bashrc = Path(path)
                print(f"   ✅ Found OpenFOAM at: {self.openfoam_bashrc}")
                return

        print(f"   ❌ Could not find OpenFOAM installation")
        print(f"   Please specify openfoam_bashrc manually")

    def run_case(self, case_dir, log_to_file=True):
        """
        Run a single OpenFOAM case through complete workflow.

        Args:
            case_dir: Path to OpenFOAM case directory
            log_to_file: Whether to log output to files

        Returns:
            dict: Results with success status and log paths
        """
        case_dir = Path(case_dir).resolve()

        if not case_dir.exists():
            raise FileNotFoundError(f"Case directory not found: {case_dir}")

        print(f"\n{'='*70}")
        print(f"🚀 Running OpenFOAM case: {case_dir.name}")
        print(f"{'='*70}")

        results = {
            'case_id': case_dir.name,
            'case_dir': str(case_dir),
            'success': False,
            'steps_completed': []
        }

        # Change to case directory
        original_dir = os.getcwd()
        os.chdir(case_dir)

        try:
            # Step 1: Run blockMesh
            if self.run_blockmesh:
                print("\n📐 Step 1: Generating background mesh (blockMesh)...")
                success = self._run_command(
                    "blockMesh",
                    log_file="log.blockMesh" if log_to_file else None
                )
                if success:
                    results['steps_completed'].append('blockMesh')
                    print("   ✅ blockMesh completed")
                else:
                    print("   ❌ blockMesh failed")
                    return results

            # Step 2: Run surfaceFeatures (extract geometry features)
            print("\n🔍 Step 2: Extracting surface features...")
            success = self._run_command(
                "surfaceFeatures",
                log_file="log.surfaceFeatures" if log_to_file else None
            )
            if success:
                results['steps_completed'].append('surfaceFeatures')
                print("   ✅ surfaceFeatures completed")
            else:
                print("   ⚠️  surfaceFeatures failed (may not be critical)")

            # Step 3: Run snappyHexMesh
            if self.run_snappyhexmesh:
                print("\n🎯 Step 3: Generating body-fitted mesh (snappyHexMesh)...")
                print("   ⏳ This may take several minutes...")
                success = self._run_command(
                    "snappyHexMesh -overwrite",
                    log_file="log.snappyHexMesh" if log_to_file else None
                )
                if success:
                    results['steps_completed'].append('snappyHexMesh')
                    print("   ✅ snappyHexMesh completed")
                else:
                    print("   ❌ snappyHexMesh failed")
                    return results

            # Step 4: Check mesh quality
            print("\n✓ Step 4: Checking mesh quality...")
            success = self._run_command(
                "checkMesh",
                log_file="log.checkMesh" if log_to_file else None
            )
            if success:
                results['steps_completed'].append('checkMesh')
                print("   ✅ checkMesh completed")
            else:
                print("   ⚠️  checkMesh reported issues (check log.checkMesh)")

            # Step 5: Run solver
            if self.run_solver:
                print(f"\n🌊 Step 5: Running CFD solver ({self.solver})...")
                print("   ⏳ This may take a long time...")

                if self.parallel and self.num_procs > 1:
                    # Decompose for parallel run
                    print(f"   🔧 Decomposing domain for {self.num_procs} processors...")
                    success = self._run_command(
                        "decomposePar",
                        log_file="log.decomposePar" if log_to_file else None
                    )
                    if not success:
                        print("   ❌ decomposePar failed")
                        return results

                    results['steps_completed'].append('decomposePar')

                    # Run solver in parallel
                    solver_cmd = f"mpirun -np {self.num_procs} {self.solver} -parallel"
                    success = self._run_command(
                        solver_cmd,
                        log_file=f"log.{self.solver}" if log_to_file else None
                    )

                    if success:
                        results['steps_completed'].append(self.solver)
                        print(f"   ✅ {self.solver} completed")

                        # Reconstruct parallel solution
                        print("   🔧 Reconstructing parallel solution...")
                        success = self._run_command(
                            "reconstructPar",
                            log_file="log.reconstructPar" if log_to_file else None
                        )
                        if success:
                            results['steps_completed'].append('reconstructPar')
                            print("   ✅ reconstructPar completed")
                    else:
                        print(f"   ❌ {self.solver} failed")
                        return results
                else:
                    # Run solver in serial
                    success = self._run_command(
                        self.solver,
                        log_file=f"log.{self.solver}" if log_to_file else None
                    )
                    if success:
                        results['steps_completed'].append(self.solver)
                        print(f"   ✅ {self.solver} completed")
                    else:
                        print(f"   ❌ {self.solver} failed")
                        return results

            # If we got here, everything succeeded
            results['success'] = True
            print(f"\n✅ Case {case_dir.name} completed successfully!")

        except Exception as e:
            print(f"\n❌ Error running case: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)

        finally:
            os.chdir(original_dir)

        return results

    def _run_command(self, command, log_file=None):
        """
        Run an OpenFOAM command with proper environment.

        Args:
            command: Command to run
            log_file: Optional log file path

        Returns:
            bool: True if command succeeded
        """
        # Build shell command with OpenFOAM environment
        shell_cmd = f"source {self.openfoam_bashrc} && {command}"

        if log_file:
            shell_cmd += f" > {log_file} 2>&1"

        # Run command
        try:
            result = subprocess.run(
                shell_cmd,
                shell=True,
                executable="/bin/bash",
                check=False,
                capture_output=(log_file is None),
                text=True
            )

            # Print output if not logging to file
            if log_file is None and result.stdout:
                print(result.stdout)

            return result.returncode == 0

        except Exception as e:
            print(f"   ❌ Error running command: {e}")
            return False

    def run_multiple_cases(self, cases_dir, case_pattern="case_*", sequential=True):
        """
        Run multiple OpenFOAM cases.

        Args:
            cases_dir: Directory containing case subdirectories
            case_pattern: Glob pattern to match case directories
            sequential: Whether to run cases sequentially (True) or in parallel (False)

        Returns:
            list: List of results dictionaries
        """
        cases_dir = Path(cases_dir)
        case_dirs = sorted(cases_dir.glob(case_pattern))

        if not case_dirs:
            print(f"No cases found matching pattern '{case_pattern}' in {cases_dir}")
            return []

        print(f"\n{'='*70}")
        print(f"🚀 Running {len(case_dirs)} OpenFOAM cases")
        print(f"{'='*70}")

        results = []

        if sequential:
            for i, case_dir in enumerate(case_dirs, 1):
                print(f"\n[{i}/{len(case_dirs)}] Processing: {case_dir.name}")
                result = self.run_case(case_dir)
                results.append(result)

                # Save intermediate results
                self._save_results(cases_dir / "simulation_results.json", results)
        else:
            # TODO: Implement parallel execution with multiprocessing or job scheduler
            print("⚠️  Parallel execution not yet implemented, falling back to sequential")
            return self.run_multiple_cases(cases_dir, case_pattern, sequential=True)

        # Print summary
        self._print_summary(results)

        return results

    def _save_results(self, output_file, results):
        """Save results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

    def _print_summary(self, results):
        """Print summary of simulation results."""
        print(f"\n{'='*70}")
        print("📊 SIMULATION SUMMARY")
        print(f"{'='*70}")

        total = len(results)
        successful = sum(1 for r in results if r['success'])
        failed = total - successful

        print(f"Total cases:      {total}")
        print(f"✅ Successful:    {successful}")
        print(f"❌ Failed:        {failed}")

        if failed > 0:
            print(f"\nFailed cases:")
            for r in results:
                if not r['success']:
                    print(f"  - {r['case_id']}: stopped at {r['steps_completed'][-1] if r['steps_completed'] else 'start'}")


def main():
    """Main function for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Run OpenFOAM cases for AAA geometries")
    parser.add_argument(
        "cases_dir",
        type=str,
        help="Directory containing OpenFOAM case subdirectories"
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="pimpleFoam",
        help="OpenFOAM solver to use (default: pimpleFoam)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run solver in parallel with MPI"
    )
    parser.add_argument(
        "--num-procs",
        type=int,
        default=1,
        help="Number of processors for parallel runs (default: 1)"
    )
    parser.add_argument(
        "--skip-blockmesh",
        action="store_true",
        help="Skip blockMesh step"
    )
    parser.add_argument(
        "--skip-snappy",
        action="store_true",
        help="Skip snappyHexMesh step"
    )
    parser.add_argument(
        "--skip-solver",
        action="store_true",
        help="Skip solver step (only mesh generation)"
    )
    parser.add_argument(
        "--openfoam-bashrc",
        type=str,
        default="/opt/openfoam9/etc/bashrc",
        help="Path to OpenFOAM bashrc"
    )

    args = parser.parse_args()

    # Create runner
    runner = OpenFOAMCaseRunner(
        openfoam_bashrc=args.openfoam_bashrc,
        solver=args.solver,
        parallel=args.parallel,
        num_procs=args.num_procs,
        run_blockmesh=not args.skip_blockmesh,
        run_snappyhexmesh=not args.skip_snappy,
        run_solver=not args.skip_solver
    )

    # Run all cases
    results = runner.run_multiple_cases(args.cases_dir)

    # Exit with appropriate code
    success = all(r['success'] for r in results)
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
