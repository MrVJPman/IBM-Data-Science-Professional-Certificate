--Query 1A: Select the names and job start dates of all employees who work for the department number 5.

SELECT F_NAME, L_NAME, START_DATE 
FROM EMPLOYEES E INNER JOIN  JOB_HISTORY JB
ON E.JOB_ID =  JB.JOBS_ID 
WHERE E.DEP_ID = 5;

--Query 1B: Select the names, job start dates, and job titles of all employees who work for the department number 5.

SELECT F_NAME, L_NAME, START_DATE,  JOB_TITLE 
FROM EMPLOYEES E 
INNER JOIN  JOB_HISTORY JB 
ON E.JOB_ID =  JB.JOBS_ID 
INNER JOIN  JOBS J 
ON JB.JOBS_ID =  J.JOB_IDENT 
WHERE E.DEP_ID = 5;

--Query 2A: Perform a Left Outer Join on the EMPLOYEES and DEPARTMENT tables and select employee id, last name, department id and department name for all employees

SELECT EMP_ID, L_NAME, E.DEP_ID,  DEP_NAME 
FROM EMPLOYEES E  LEFT OUTER JOIN  DEPARTMENTS D 
ON E.DEP_ID =  D.DEPT_ID_DEP ;

--Query 2B: Re-write the query for 2A to limit the result set to include only the rows for employees born before 1980.

SELECT EMP_ID, L_NAME, E.DEP_ID,  DEP_NAME 
FROM EMPLOYEES E 
INNER JOIN  DEPARTMENTS D 
ON E.DEP_ID =  D.DEPT_ID_DEP 
WHERE YEAR(E.B_DATE) < 1980;

--Query 2C: Re-write the query for 2A to have the result set include all the employees but department names for only the employees who were born before 1980.

SELECT EMP_ID, L_NAME, E.DEP_ID,  DEP_NAME 
FROM EMPLOYEES E  LEFT OUTER JOIN  DEPARTMENTS D 
ON E.DEP_ID =  D.DEPT_ID_DEP AND YEAR(E.B_DATE) < 1980;

--Query 3A: Perform a Full Join on the EMPLOYEES and DEPARTMENT tables and select the First name, Last name and Department name of all employees.

SELECT F_NAME, L_NAME,  DEP_NAME 
FROM EMPLOYEES E  FULL OUTER JOIN  DEPARTMENTS D 
ON E.DEP_ID =  D.DEPT_ID_DEP ;

--Query 3B: Re-write Query 3A to have the result set include all employee
--names but department id and department names only for male employees.

SELECT F_NAME, L_NAME,  DEP_NAME, DEP_ID 
FROM EMPLOYEES E FULL OUTER JOIN  DEPARTMENTS D 
ON E.DEP_ID =  D.DEPT_ID_DEP AND E.SEX = 'M';
