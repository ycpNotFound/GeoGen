import json
import os



def split_data(image_dir, json_path):
    json_data = json.load(open(json_path, 'r', encoding='utf-8'))
    
    
def main():
    image_dir = "geo_gen/geo_gen_pretrain/imgs"
    split_data()
    
    
    
if __name__ == '__main__':
    main()