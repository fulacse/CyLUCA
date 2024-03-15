CyPred v 1.00 

Cyclic proteins (CPs) have circular chains with a continuous cycle of 
peptide bonds. Their unique structural traits give them greater 
stability, better receptor selectively, and improved pharmacodynamic 
properties when compared to their acyclic counterparts, making them 
promising targets for pharmaceutical/therapeutic applications. This 
package includes first-of-its-kind, high-throughput sequence-based
predictor of CPs, called CyPred. 

============= 

Usage 

java -jar CyPred.jar inputFile outputFile 

CyPred accepts either single or multiple protein sequences. The user 
should prepare the protein sequence(s) in FASTA format (inputFile). 

The format of the input file is as follow (example,incomplete proteome 
of Violaceae, is provided in ./example/Violaceae.fasta): 

>protein name
protein sequence (one letter amino acid code only) 

The ouputfile will have following format: 

>protein name predicted score - the higher score the more likely a 
protein is cyclic (positive score - cyclic, negative score - not cyclic) 

Example usage: 

java -jar CyPred.jar ./example/Violaceae.fasta 
./example/Violaceae.cypred 

Do not remove files from lib folder or CyPred will not execute! CyPred 
should execute on any operating system where JAVA (version 6 or higher) 
is installed. To install JAVA visit: http://www.java.com/

Files tree: ./lib/cyclicLib.jar ./lib/CyPred.model 
./example/Violaceae.fasta ./CyPred.jar ./README 

============= 

References 

Upon the usage the users are requested to use the following citation: 

Kedarisetti P, Mizianty MJ, Kaas Q, Craik DJ, Kurgan L. Prediction and 
characterization of cyclic proteins from sequences in three domains of 
life. Submitted 

============= 

Acknowledgments 

We acknowledge, with thanks, that the following software was used as a 
part of this program: 

Weka 3 - Data Mining Software in Java 
(http://www.cs.waikato.ac.nz/ml/weka/) 

