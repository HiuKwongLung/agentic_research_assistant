import argparse
from agent.graph import run_agent

def main():
    parser = argparse.ArgumentParser(
        description="Agentic Research Assistant - give a topic, and the agent will research it and write a report."
    )
    parser.add_argument(
        "topic",
        type=str,
        help="The research topic to investigate."
    )

    args = parser.parse_args()

    print(f"\nResearching: {args.topic}\n")
    print("-" * 50)

    result = run_agent(args.topic)

    print("\n" + result)
    print("-" * 50)
    print ("Done! Check the reports/ folder for the saved report.")

if __name__ == "__main__":
    main()