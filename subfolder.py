import os

origin_folder = 'data/adjacent_matrix/pm0.3'
origin_files = os.listdir(origin_folder)

for origin_file in origin_files:
    folder = 'data/adjacent_matrix/pm0.3/' + origin_file
    print(folder)
    files = os.listdir(folder)
    
    for sub_folder in files:
        sub_folder_path = os.path.join(folder, sub_folder)
        if os.path.isdir(sub_folder_path):
            sub_folder_files = os.listdir(sub_folder_path)
            for i, file in enumerate(sub_folder_files):
                if i >= 7:
                    file_path = os.path.join(sub_folder_path, file)
                    os.remove(file_path)