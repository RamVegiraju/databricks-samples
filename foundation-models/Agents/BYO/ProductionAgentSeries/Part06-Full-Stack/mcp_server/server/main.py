import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start the MCP server")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(
        "server.app:combined_app",
        host="0.0.0.0",
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
