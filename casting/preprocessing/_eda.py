
# 불량률 계산 함수
def calculate_defect_rate(df, lower_bound, upper_bound, x_name):
    filtered_data = df[(df[x_name] > lower_bound) & (df[x_name] <= upper_bound)]
    a = len(filtered_data[filtered_data["passorfail"] == 1])
    b = len(filtered_data[filtered_data['passorfail'] != 1])
    return (a / (a + b)) * 100 if (a + b) > 0 else 0
