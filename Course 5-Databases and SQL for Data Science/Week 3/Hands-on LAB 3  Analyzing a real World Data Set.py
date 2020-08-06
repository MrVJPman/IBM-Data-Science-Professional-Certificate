How many rows are in the dataset?

%sql SELECT count(*) FROM chicago_socioeconomic_data

How many community areas in Chicago have a hardship index greater than 50.0?

%sql SELECT count(*) FROM chicago_socioeconomic_data WHERE hardship_index > 50.0

What is the maximum value of hardship index in this dataset?

%sql SELECT MAX(hardship_index) FROM chicago_socioeconomic_data

Which community area which has the highest hardship index?

%sql SELECT community_area_name FROM chicago_socioeconomic_data WHERE hardship_index = (SELECT MAX(hardship_index) FROM chicago_socioeconomic_data)

Which Chicago community areas have per-capita incomes greater than $60,000?

%sql SELECT community_area_name FROM chicago_socioeconomic_data WHERE per_capita_income_ > 60000	