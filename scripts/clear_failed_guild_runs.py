import argparse
import os
import shutil


class RunRemover:
    invalid_count = 0
    delete_count = 0

    is_dry_run: bool

    def run(self, args):
        self.is_dry_run = args.dry_run

        run_directories = [
            f.path for f in os.scandir(args.guild_runs_path) if f.is_dir()
        ]

        for run_dir in run_directories:
            attrs_dir = os.path.join(run_dir, ".guild", "attrs")
            if not os.path.exists(attrs_dir):
                print(f"invalid run: {run_dir}")
                self.invalid_count += 1
                continue
            started_filepath = os.path.join(attrs_dir, "started")
            stopped_filepath = os.path.join(attrs_dir, "stopped")
            if not os.path.exists(started_filepath) or not os.path.exists(
                stopped_filepath
            ):
                self.delete_run(run_dir)
                continue

            started_time, stopped_time = None, None
            with open(started_filepath) as f:
                s = f.read()
                started_time = int(s)
            with open(stopped_filepath) as f:
                s = f.read()
                stopped_time = int(s)

            mirosec_difference = stopped_time - started_time
            if mirosec_difference < 0:
                print(f"invalid run time: {mirosec_difference}, {run_dir}")
                self.invalid_count += 1
                continue

            sec_difference = mirosec_difference / 1000000
            print(sec_difference, run_dir)
            if sec_difference < args.run_time_threshold:
                self.delete_run(run_dir)

        print(f"{self.invalid_count = }")
        print(f"{self.delete_count = }")
        if self.is_dry_run:
            print("This was a dry run. No files were actually deleted.")

    def delete_run(self, run_dir: str) -> None:
        print(f"deleting {run_dir = }")
        self.delete_count += 1
        if not self.is_dry_run:
            shutil.rmtree(run_dir)

    def add_argparse_arguments(
        self, parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        parser.add_argument(
            "--guild_runs_path",
            type=str,
        )
        parser.add_argument(
            "--run_time_threshold",
            type=int,
            default=120,
            help="Run time in seconds. If a run took shorter than this time it will get deleted.",
        )
        parser.add_argument(
            "--dry_run",
            action="store_true",
            help="Do not actually delete. Show the results only.",
        )
        return parser


if __name__ == "__main__":
    run_remover = RunRemover()

    parser = argparse.ArgumentParser()
    parser = run_remover.add_argparse_arguments(parser)
    args = parser.parse_args()

    run_remover.run(args)
