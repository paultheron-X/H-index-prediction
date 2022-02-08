import json
import string
import progress_bar as pb
import argparse
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords



parser = argparse.ArgumentParser(description="Re-build abstracts from their index")
parser.add_argument("input_file")
parser.add_argument("output_file")

args = parser.parse_args()

input_file_name = args.input_file
output_file_name = args.output_file

def compute_line(json_line, lower_case=True, remove_punctuation=True, remove_positionners=True, remove_stopwords= True):
    # Builds the reconstructed array
    words = json_line["InvertedIndex"]

    result_array = [None] * json_line["IndexLength"]
    for word, indexes in words.items():
        for index in indexes:
            result_array[index] = word
    # Builds a string from the array
    result = ""
    for word in result_array:
        if (word == None):
            continue
        result += " " + word
    if (lower_case):
        result = result.lower()
    if (remove_punctuation):
        result = result.translate(str.maketrans('', '', string.punctuation))
    if (remove_stopwords):
        stop_words = set(stopwords.words('english'))
        stop_words = [" " + word + " " for word in stop_words]
        for word in stop_words:
            result = result.replace(word, " ")
    if remove_positionners:
        result = result.replace("\n", "")
        result = result.replace("\p", "")
        result = result.replace("\r", "")
        result = result.replace("’", "")
        result = result.replace("‘", "")
    result = result[1:] # '1:' to remove initial space
    return result

pb.init(1, _prefix = 'Building abstracts \t \t', _suffix = 'Complete')
pb.progress(0)

input_file = open(input_file_name, 'r', encoding='UTF8')
output_file = open(output_file_name, 'w', encoding='UTF8')

lines = input_file.readlines()
line_count = len(lines)
count = 0

# Init progress bar
pb.set_length(line_count - 1)

output_file.write("ID;abstract\n")
for line in lines:
    split = line.split('----', 1)
    json_line = json.loads(split[1])
    result = compute_line(json_line)
    result =  split[0] + ";" + result + "\n"
    output_file.write(result)
    
    pb.progress(count)
    count = count + 1

input_file.close()
output_file.close()
