import argparse
import progress_bar as pb

parser = argparse.ArgumentParser(description="Formats the list of publication per author into a csv file")
parser.add_argument("input_file", help="Input .txt file of the publications per author")
parser.add_argument("output_file", help="Output .csv file")
parser.add_argument("output_file_count", help="Output .csv file")

args = parser.parse_args()
    
input_file_name = args.input_file
output_file_name = args.output_file
output_file_count_name = args.output_file_count


pb.init(1, _prefix="Formating authors \t \t")
pb.progress(0)

input_file = open(input_file_name, 'r', encoding='UTF8')
output_file = open(output_file_name, 'w', encoding='UTF8')
output_file_count = open(output_file_count_name, 'w', encoding='UTF8')

lines = input_file.readlines()
line_count = len(lines)
count = 0

pb.set_length(line_count - 1)

output_file.write("author;paper_1;paper_2;paper_3;paper_4;paper_5\n")
output_file_count.write("author;count\n")

for line in lines:
    author = line.split(":", 1)[0]
    # print(author)

    line = line.replace(":", ";")
    line = line.replace("-", ";")
    line = line.replace("\n", "")
    nb_publications = line.count(";")
    line = line + (';' * (5 - nb_publications))
    line = line + "\n"
    output_file.write(line)
    output_file_count.write(str(author) + ";" + str(nb_publications) + "\n")

    pb.progress(count)
    count = count + 1

input_file.close()
output_file.close()
output_file_count.close()
