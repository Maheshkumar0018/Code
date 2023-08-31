
# read categories into list formate
category_file_path = './category.txt'  
names_list = []
with open(category_file_path, 'r') as file:
    # Skip the header (first line with 'id name')
    next(file)
    # Extract names from the file and add them to the list
    for line in file:
        _, name = line.strip().split('\t')
        names_list.append(name)

print(names_list)
print(len(names_list))

# read categoreis into dictionary formate 
def read_categories_file(file_path):
    category_mapping = {}
    with open(file_path, "r") as file:
        header = next(file)  # Skip the header line
        for line in file:
            line = line.strip()
            if line:
                category_id, category_name = line.split('\t')
                category_mapping[int(category_id)] = category_name.strip()

    return category_mapping




