import subprocess
import argparse
import time
from types import SimpleNamespace
import sys

"""terminal color"""
TC = SimpleNamespace(
    **{
        "YELLOW": "\033[33m",
        "GREEN": "\033[92m",
        "RED": "\033[91m",
        "BLUE": "\033[34m",
        "RESET": "\033[0m",
    }
)


class Cmd:
    def __new__(
        self, cmd: str, cwd="./", timeout_duration=None, suppress=False
    ) -> tuple[int, str, str]:
        self.cmd = cmd
        self.cwd = cwd
        self.returncode = 0
        self.has_err = True

        if not suppress:
            print(f"{self.cmd}", end="", flush=True)
        cwd_not_cur = f" in {self.cwd}" if self.cwd != "./" else ""

        """ process setup """
        process = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            executable="bash",
            cwd=self.cwd,
        )

        """ timeout """
        # https://stackoverflow.com/a/13821695
        import signal

        class TimeoutError(Exception):
            pass

        def handler(signum, frame):
            raise TimeoutError()

        # set the timeout handler
        if timeout_duration is not None:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout_duration)

        """ execution """
        out = bytearray()
        err = bytearray()
        timeStarted = time.time()
        try:
            _out, _err = process.communicate()
            out = _out if _out is not None else out
            err = _err if _err is not None else err
            self.returncode = process.returncode
            if process.returncode != 0:
                raise RuntimeError(
                    f"returncode is not 0 but {process.returncode}. "
                    + str(out + err, encoding="utf8")
                )
        except RuntimeError as e:
            if not suppress:
                print(f"{cwd_not_cur} {TC.RED}[failed]{TC.RESET}")
            return self.returncode, str(out, encoding="utf8"), str(err, encoding="utf8")
        except TimeoutError as e:
            if not suppress:
                print(f"{cwd_not_cur} {TC.RED}[failed]{TC.RESET}")
            return self.returncode, str(out, encoding="utf8"), str(err, encoding="utf8")
        except:
            if not suppress:
                print(f"{cwd_not_cur} {TC.RED}[failed]{TC.RESET}")
            return self.returncode, str(out, encoding="utf8"), str(err, encoding="utf8")
        finally:  # reset timeout handler
            signal.alarm(0)

        timeDelta = time.time() - timeStarted
        if not suppress:
            print(f"{cwd_not_cur} {TC.GREEN}[passed]{TC.RESET} ({timeDelta:.3f}s)")
        self.has_err = False
        return self.returncode, str(out, encoding="utf8"), str(err, encoding="utf8")


def list_of_strings(arg):
    return arg.split(",")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnodes", type=int)
    parser.add_argument("--node-rank", type=int)
    parser.add_argument("--nproc-per-node", type=int)
    parser.add_argument("--master-addr", type=str)
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--target-script", type=str)
    parser.add_argument("--terminate", action="store_true")
    args = parser.parse_args()

    Cmd("""tmux kill-session -t merlin || true""")
    if args.terminate:
        return

    rc, out, err = Cmd("tmux -f /dev/null new-session -s merlin -d bash\;")

    if rc != 0:
        print(err, file=sys.stderr)
        sys.exit(1)

    Cmd("tmux set-option -g mouse on")
    Cmd("tmux send-keys -t 0 'clear' Enter \;")
    Cmd("tmux send-keys -t 0 'conda activate merlin' Enter \;")
    Cmd(
        "tmux send-keys -t 0 'torchrun "
        + f"--nnodes={args.nnodes} "
        + f"--node-rank={args.node_rank} "
        + f"--nproc-per-node={args.nproc_per_node} "
        + f"--master-addr={args.master_addr} "
        + f"--master-port={args.master_port} "
        + f"{args.target_script}' Enter \;"
    )


if __name__ == "__main__":
    main()
