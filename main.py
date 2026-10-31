"""Main entrypoint for HANDVISION STUDIO — REALITYFRAME EDITION.

Parses command-line arguments, sets up logging, configures the application,
and launches the interactive OpenCV desktop window.
"""

import argparse
import logging
import sys

from config import (
    AppConfig,
    AppMode,
    CameraConfig,
    FilterConfig,
    HandTrackerConfig,
    KeyboardConfig,
    RealityFrameConfig,
    ROIConfig,
)
from src.application import HandFilterApp


def setup_logging(verbose: bool = False) -> None:
    """Configure console logging with timestamps and level formatting."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HANDVISION STUDIO — REALITYFRAME EDITION: Real-Time Hand-Controlled Vision, Ghost Portal & Air Keyboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam device index (0 is default, 1 or 2 for external webcams).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested capture width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested capture height in pixels.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="STANDARD_FILTER",
        choices=[
            "STANDARD_FILTER",
            "REALITYFRAME_ROI",
            "REALITYFRAME_GHOST",
            "VIRTUAL_KEYBOARD",
            "AIR_WRITING",
            "MYSTIC_MAGIC",
            "UNIVERSE_PORTAL",
        ],
        help="Initial application operating mode.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="negative",
        choices=["negative", "canny", "thermal", "sketch", "grayscale", "pixelated", "custom"],
        help="Initial active visual filter.",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="INDEX_FINGERTIP",
        choices=["INDEX_FINGERTIP", "INDEX_MCP", "WRIST"],
        help="Hand landmark anchor used for corner control.",
    )
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="Start directly in Virtual Keyboard mode.",
    )
    parser.add_argument(
        "--air-writing",
        action="store_true",
        help="Start directly in Air Writing Studio mode.",
    )
    parser.add_argument(
        "--magic",
        action="store_true",
        help="Start directly in Mystic Magic Engine mode.",
    )
    parser.add_argument(
        "--universe",
        action="store_true",
        help="Start directly in Cinematic Universe Portal mode.",
    )
    parser.add_argument(
        "--typing-mode",
        type=str,
        default="PINCH",
        choices=["PINCH", "DWELL"],
        help="Typing trigger method: PINCH (thumb+index) or DWELL (hold finger on key).",
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable exponential coordinate smoothing for minimum latency.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    """Construct an AppConfig instance from CLI arguments."""
    cam_config = CameraConfig(
        camera_index=args.camera_index,
        requested_width=args.width,
        requested_height=args.height,
    )
    tracker_config = HandTrackerConfig(
        landmark_anchor=args.anchor,
    )
    roi_config = ROIConfig(
        smoothing_enabled=not args.no_smoothing,
    )
    filter_config = FilterConfig(
        default_filter=args.filter,
    )
    keyboard_config = KeyboardConfig(
        enabled=args.keyboard or (args.mode == "VIRTUAL_KEYBOARD"),
        typing_mode=args.typing_mode,
    )

    initial_mode = AppMode[args.mode]
    if args.keyboard:
        initial_mode = AppMode.VIRTUAL_KEYBOARD
    elif args.air_writing:
        initial_mode = AppMode.AIR_WRITING
    elif args.magic:
        initial_mode = AppMode.MYSTIC_MAGIC
    elif args.universe:
        initial_mode = AppMode.UNIVERSE_PORTAL

    return AppConfig(
        camera=cam_config,
        hand_tracker=tracker_config,
        roi=roi_config,
        reality_frame=RealityFrameConfig(),
        filters=filter_config,
        keyboard=keyboard_config,
        initial_mode=initial_mode,
    )


def main() -> int:
    """Main execution function."""
    args = parse_arguments()
    setup_logging(args.verbose)

    config = build_config(args)
    config.validate()

    print("=================================================================")
    print(" HANDVISION STUDIO — REALITYFRAME: CINEMATIC UNIVERSE PORTAL     ")
    print("=================================================================")
    print(f" Initial Mode:    {config.initial_mode.value}")
    print(f" Camera Index:    {config.camera.camera_index}")
    print(f" Target Res:      {config.camera.requested_width}x{config.camera.requested_height}")
    print(f" Active Filter:   {config.filters.default_filter}")
    print(f" Typing Mode:     {config.keyboard.typing_mode}")
    print(f" Landmark Anchor: {config.hand_tracker.landmark_anchor}")
    print(f" ROI Smoothing:   {'Enabled' if config.roi.smoothing_enabled else 'Disabled'}")
    print("-----------------------------------------------------------------")
    print(" Modes:")
    print("   [1] - Standard Filter Box (Two-Hand ROI)")
    print("   [2] - RealityFrame ROI Portal (Cybernetic Filter)")
    print("   [3] - RealityFrame Ghost Portal (Transparent Optical Illusion)")
    print("   [4] or [K] - Virtual Finger Typing Air Keyboard")
    print("   [5] or [W] - Air Writing Studio (Draw & Sketch in Air)")
    print("   [6] or [G] - Mystic Magic Engine (Doctor Strange Spells)")
    print("   [7] or [U] - Cinematic Universe Portal (3D Solar System & Stars)")
    print(" Shortcut Controls:")
    print("   [R] - Recapture RealityFrame Background Plate")
    print("   [N] / [P] - Next / Previous Visual Filter")
    print("   [X] - Toggle Cinematic / Scientific Planet Scale")
    print("   [SPACE] - Pause / Resume 3D Universe Orbit Simulation")
    print("   [Z] / [Y] - Undo / Redo Drawn Stroke")
    print("   [C] - Clear Canvas / Clear Typed Text / Close Universe Portal")
    print("   [S] - Save Air Writing Drawing to PNG")
    print("   [G] - Toggle Magical Glow / Cycle Mystic Spells")
    print("   [F] / [H] - Toggle HUD Telemetry Dashboard")
    print("   [Q] / [ESC] - Exit Application")
    print(" Mouse Click:")
    print("   Click any button on the top Dashboard to switch modes or trigger actions!")
    print("=================================================================\n")

    app = HandFilterApp(config)
    try:
        app.run()
        return 0
    except Exception as e:
        logging.exception("Fatal application error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
