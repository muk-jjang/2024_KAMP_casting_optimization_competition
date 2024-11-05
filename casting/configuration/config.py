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
    
    context_columns = ['facility_operation_cycleTime', 'production_cycletime', 'low_section_speed', 'high_section_speed', 
                       'biscuit_thickness', 'upper_mold_temp1', 'upper_mold_temp3', 'lower_mold_temp1', 'lower_mold_temp3',
                         'sleeve_temperature', 'physical_strength', 'Coolant_temperature', 'EMS_operation_time',
                  'mold_code', 'molten_temp']
    
    action_columns = ['upper_mold_temp2', 'lower_mold_temp2', 'cast_pressure']

class paths:
    origin_path = 'casting/data/raw/casting_data_origin.csv'

    ml_train_path = 'casting/data/processed_data/train.csv'
    ml_valid_path = 'casting/data/processed_data/valid.csv'
    ml_test_path = 'casting/data/processed_data/test.csv'

    dl_train_path = 'casting/data/scaled_data/train.csv'
    dl_valid_path = 'casting/data/scaled_data/valid.csv'
    dl_test_path = 'casting/data/scaled_data/test.csv'
    dl_all_path = 'casting/data/scaled_data/all.csv'

    X_scaler_path = 'casting/data/scaler/X_scaler.pickle'
    y_scaler_path = 'casting/data/scaler/y_scaler.pickle'

    label_encoding_path = 'casting/data/scaler/label_encoding.pickle'

    FTT_path = 'casting/result/predictor_model/FTT'


class ranges :
    outlier_set = {
    'molten_temp' : [0, 735],
    'facility_operation_cycleTime' : [69, 268],
    'production_cycletime' : [0, 360],
    'low_section_speed' : [0, 160],
    'high_section_speed' : [0, 340],
    'cast_pressure' : [41, 344],
    'biscuit_thickness' : [0, 422],
    'upper_mold_temp1' : [18, 332],
    'upper_mold_temp2' : [15, 241],
    'upper_mold_temp3' : [42, 1449],
    'lower_mold_temp1' : [18, 367],
    'lower_mold_temp2' : [20, 492],
    'lower_mold_temp3' : [299, 1449],
    'sleeve_temperature' : [23, 1449],
    'physical_strength' : [0 ,736],
    'Coolant_temperature' : [16, 49],
    }

class params :
    # model_name = 'FTT'
    # model_name = 'TABNET'
    # model_name = 'XGB'
    model_name = 'EXT'
    # model_name = 'LGBM'
    
class optimization:
    """Configuration for the Bayesian optimization process."""
    param_bounds: param_bounds={
        'cast_pressure': (40, 380),
        'lower_mold_temp2': (20, 540),
        'upper_mold_temp2': (15, 270)
    }
    init_points: int = 5
    n_iter: int = 10
    n_processes: int = 16
    verbose: int = 0
