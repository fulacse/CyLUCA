import subprocess
import tempfile
import os


def cypred(sequence: str) -> float:
    cwd = os.getcwd()
    os.chdir("tools/CyPred")

    # Create a temporary FASTA file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta') as temp_fasta:
        fasta_filename = temp_fasta.name
        # Write the input sequence to the temporary FASTA file
        temp_fasta.write(f">temp_sequence\n{sequence}\n")

    # Define the output file name
    output_filename = f"{fasta_filename}.cypred"

    # Execute the CyPred Java application
    try:
        subprocess.run(['java', '-jar', 'CyPred.jar', fasta_filename, output_filename], check=True)

        # Read the predicted score from the output file
        with open(output_filename, 'r') as output_file:
            return float(output_file.read().split()[1])
    finally:
        # Clean up temporary files
        os.remove(fasta_filename)
        os.remove(output_filename)
        os.chdir(cwd)


if __name__ == "__main__":
    # Example usage
    score = cypred("GIPCGESCVYIPCLTSAIGCSCKSKVCYRN")
    print(f"Predicted score: {score}")
