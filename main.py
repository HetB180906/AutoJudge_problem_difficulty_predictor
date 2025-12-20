from src.model import full_pred
def main():
    print("=== Predicting Problem difficulty (class) and score ===\n")
    text=input("\nEnter problem statement:\n> ")
    result=full_pred(text)
    print("\nPrediction Result:")
    print(f"Difficulty: {result['difficulty']}")
    print(f"Score     : {result['predicted_score']}")

if __name__ == "__main__":
    main()
