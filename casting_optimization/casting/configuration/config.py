class columns:
    use_columns = [ 'molten_temp', 'facility_operation_cycleTime', 'production_cycletime', 'low_section_speed', 'high_section_speed',
                 'cast_pressure', 'biscuit_thickness', 'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3', 'lower_mold_temp1',
                'lower_mold_temp2', 'lower_mold_temp3', 'sleeve_temperature', 'physical_strength', 'Coolant_temperature', 'EMS_operation_time',
                  'mold_code', 'passorfail']

    category_columns = ['mold_code']
    numeric_columns = ['molten_temp', 'facility_operation_cycleTime', 'production_cycletime', 'low_section_speed', 'high_section_speed',
                 'cast_pressure', 'biscuit_thickness', 'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3', 'lower_mold_temp1',
                'lower_mold_temp2', 'lower_mold_temp3', 'sleeve_temperature', 'physical_strength', 'Coolant_temperature', 'EMS_operation_time']
    target_column = ['passorfail']

    input_columns = ['molten_temp', 'facility_operation_cycleTime', 'production_cycletime', 'low_section_speed', 'high_section_speed',
                 'cast_pressure', 'biscuit_thickness', 'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3', 'lower_mold_temp1',
                'lower_mold_temp2', 'lower_mold_temp3', 'sleeve_temperature', 'physical_strength', 'Coolant_temperature', 'EMS_operation_time',
                  'mold_code']

class paths:
    origin_path = 'casting/data/raw/casting_data_origin.csv'

    ml_train_path = 'casting/data/processed_data/train.csv'
    ml_valid_path = 'casting/data/processed_data/valid.csv'
    ml_test_path = 'casting/data/processed_data/test.csv'

    dl_train_path = 'casting/data/scaled_data/train.csv'
    dl_valid_path = 'casting/data/scaled_data/valid.csv'
    dl_test_path = 'casting/data/scaled_data/test.csv'

    X_scaler_path = 'casting/data/scaler/X_scaler.pickle'
    y_scaler_path = 'casting/data/scaler/y_scaler.pickle'

    label_encoding_path = 'casting/data/scaler/label_encoding.pickle'


class ranges :
    outlier_set = {
    'biscuit_thickness' : [0, 100],
    'low_section_speed' : [0, 160],
    'upper_mold_temp1' : [0, 380],
    'upper_mold_temp2' : [0, 390],
    'lower_mold_temp2' : [0, 510],
    'lower_mold_temp3' : [290, 1500],
    'physical_strength' : [0 ,740],
    'Coolant_temperature' : [15, 60],
    }

class params :
    model_name = 'FTT'
    

