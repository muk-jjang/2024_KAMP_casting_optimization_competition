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