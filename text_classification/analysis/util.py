import pandas as pd

def get_parser_data(filename):
    """Method for reading columns of .csv file"""
    file = pd.read_csv('../' + filename, sep='\t')
    return file
