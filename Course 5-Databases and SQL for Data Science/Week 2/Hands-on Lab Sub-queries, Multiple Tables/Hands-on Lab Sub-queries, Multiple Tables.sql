--List of employees who earn more than the average salary version 1
select * from employees 
	WHERE salary  (SELECT  avg(salary) FROM employees);

--List of employees who earn more than the average salary version 2
select *, (SELECT  avg(salary) FROM employees) as avg_salary FROM employees;   

--Creating a new table to use
select * FROM (SELECT  EMP_ID, F_NAME, L_NAME, DEP_ID  FROM employees) as newtable;   
 
--Retrieve only the employee records that corresponds to departments in the DEPARTMENTS table
SELECT * FROM employees WHERE DEP_ID in (SELECT DEPT_ID_DEP FROM departments);

--Retrieve department ID and name who earn more than $70000 Method 1
SELECT DEPT_ID_DEP, DEP_NAME FROM departments where DEPT_ID_DEP IN
	(SELECT DEP_ID FROM employees WHERE SALARY  70000);

--This is a full join
--Department id for each employee
SELECT E.EMP_ID, D.DEP_ID_DEP FROM employees E, departments D where E.DEP_ID = D.DEPT_ID_DEP;
