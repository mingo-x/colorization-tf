import json

def filter_region_description():
    regions = json.load(open('/srv/gluserfs/xieya/data/visual_genome/region_descriptions.json', 'r'))
    new_regions = {}
