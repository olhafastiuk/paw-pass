"""
Single entry point for the project.

    python main.py camera --model yolo     run the door
    python main.py train                   train the classifier
    python main.py --help                  list every command

Each command imports its own module only when it runs, so starting the
camera never loads the training code, and `--help` works without torch.
"""
import argparse


def cmd_camera(args):
    from src.runtime.camera import run_camera
    run_camera(args.model, args.camera, headless=args.headless, log=not args.no_log)


def cmd_eda(args):
    from src.training.eda import main
    main(show=not args.no_show)


def cmd_model(args):
    from src.ml.model import summary
    summary()


def cmd_train(args):
    from src.training.train import train
    train()


def cmd_evaluate(args):
    from src.training.evaluate import evaluate
    evaluate(show=not args.no_show)


def cmd_errors(args):
    from src.training.error_analysis import error_analysis
    error_analysis(show=not args.no_show)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Cat door: detect a cat on camera and open the door.",
    )
    commands = parser.add_subparsers(title="commands", metavar="<command>")

    # --- runtime ---
    camera = commands.add_parser("camera", help="run the door on a live camera")
    camera.add_argument("--model", choices=["mobilenet", "yolo"], default="mobilenet",
                        help="detection backend (default: mobilenet)")
    camera.add_argument("--camera", type=int, default=0,
                        help="camera index, 0 = built-in (default: 0)")
    camera.add_argument("--headless", action="store_true",
                        help="no preview window")
    camera.add_argument("--no-log", action="store_true",
                        help="do not record door events to logs/")
    camera.set_defaults(func=cmd_camera)

    # --- training, in the order they are usually run ---
    for name, func, help_text in [
        ("eda", cmd_eda, "explore the dataset (downloads it on first run)"),
        ("model", cmd_model, "print the architecture's shapes and parameter count"),
        ("train", cmd_train, "train the classifier -> models/cat_detector.pth"),
        ("evaluate", cmd_evaluate, "threshold sweep on validation, final score on test"),
        ("errors", cmd_errors, "save FP / FN / TP examples from validation"),
    ]:
        sub = commands.add_parser(name, help=help_text)
        if name in ("eda", "evaluate", "errors"):
            sub.add_argument("--no-show", action="store_true",
                             help="save the plots without opening a window")
        sub.set_defaults(func=func)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
