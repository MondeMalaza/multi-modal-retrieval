import argparse
from src.web_app import app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--web", action="store_true", help="Run the web UI instead of CLI")
    args = parser.parse_args()

    if args.web:
        app.run(debug=True)
    else:
        from src.main import main
        main()
