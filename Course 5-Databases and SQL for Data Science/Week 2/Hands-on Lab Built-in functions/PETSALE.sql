-- Drop the PETSALE table in case it exists
drop table PETSALE;
-- Create the PETSALE table 
create table PETSALE (
	ID INTEGER PRIMARY KEY NOT NULL,
	ANIMAL VARCHAR(20),
	QUANTITY INTEGER,
	SALEPRICE DECIMAL(6,2),
	SALEDATE DATE
	);
-- Insert saple data into PETSALE table
insert into PETSALE values 
	(1,'Cat',9,450.09,'2018-05-29'),
	(2,'Dog',3,666.66,'2018-06-01'),
	(3,'Dog',1,100.00,'2018-06-04'),
	(4,'Parrot',2,50.00,'2018-06-04'),
	(5,'Dog',1,75.75,'2018-06-10'),
	(6,'Hamster',6,60.60,'2018-06-11'),
	(7,'Cat',1,44.44,'2018-06-11'),
	(8,'Goldfish',24,48.48,'2018-06-14'),
	(9,'Dog',2,222.22,'2018-06-15')
	
;

--Add all up all the values in the SALEPRICE column and name it as SUM_OF_SALEPRICE
SELECT  sum(saleprice) AS sum_of_saleprice FROM petsale;

--Max quantity from any animal
SELECT max(quantity) FROM petsale;

--Min quantity from dogs
SELECT min(ID) FROM petsale WHERE animal = 'Dog';

--Average saleprice per dog 
SELECT avg(saleprice / quantity) FROM petsale WHERE animal = 'Dog';

--Round up/down every value in SALEPRICE
SELECT round(saleprice) FROM petsale;

--Length of each ANIMAL name
SELECT length(animal) FROM petsale;

--Uppercase
SELECT UCASE(animal) FROM petsale;

--Booleans involving lower case
SELECT * FROM petsale WHERE LCASE(animal) = 'cat';

--Distinct to get unique values
SELECT DISTINCT(UCASE(animal)) FROM petsale;

--Extract day portion from a date
SELECT DAY(saledate) FROM petsale WHERE animal ='Cat';

--Number of sales during month of may
SELECT count(*) FROM petsale WHERE MONTH(saledate) = '05';

--Three days after each sale date
SELECT (saledate + 3 DAYS) FROM petsale;

--Days passed since each sale date
--CURRENT_TIME
SELECT (CURRENT_DATE - saledate) FROM petsale;