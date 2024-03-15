def parse_ss2_file(filename):
    """
    Parse a .ss2 file from PSIPRED to extract secondary structure predictions.

    Args:
        filename (str): Path to the .ss2 file.

    Returns:
        list of tuples: Each tuple contains (residue number, amino acid, predicted structure, confidence scores).
    """
    predictions = []

    with open(filename, 'r') as file:
        for line in file:
            # Skip header lines and empty lines
            if line.startswith("#") or not line.strip():
                continue

            # Split the line into components
            parts = line.split()

            # Extract relevant information
            if len(parts) >= 6:  # Ensure the line has enough parts
                residue_number = int(parts[0])
                amino_acid = parts[1]
                predicted_structure = parts[2]
                confidence_scores = tuple(float(score) for score in parts[3:6])

                predictions.append((residue_number, amino_acid, predicted_structure, confidence_scores))

    return predictions

if __name__ == "__main__":
    # Example usage
    filename = "../data/5D8VA/20f6ee26-e2ab-11ee-905a-00163e100d53.ss2"  # Update this path to your .ss2 file
    predictions = parse_ss2_file(filename)

    # Print the first few predictions
    for prediction in predictions[:5]:
        print(prediction)
