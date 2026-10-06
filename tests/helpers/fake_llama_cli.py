import sys

def main() -> int:
    if "--fail-vulkan" in sys.argv:
        print("VK_ERROR_DEVICE_LOST: GPU fence timeout", file=sys.stderr)
        return 1

    print("> assistant")
    print("This is a verified test subprocess response.")
    print("[Generation: 12.5 t/s]")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
