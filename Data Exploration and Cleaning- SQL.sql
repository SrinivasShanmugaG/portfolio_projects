-- PostgreSQL 

-- Common Table Expression

-- Without Hierarchy
with region_salaries as
 (select
    e.region_id,
    sum(e.salary) region_salary
  from
    db_test.employees e
  group by region_id),
 top_region_salaries as
  (select
      region_id
   from
      region_salaries
    where region_salary > (SELECT SUM(region_salary)/7 from region_salaries))
select 
   *
from
  region_salaries
where
  region_id in (SELECT region_id from top_region_salaries)
  
-- Recursive approach

with recursive report_structure(id, department_name,parent_department_id) as
 (select id,  department_name, parent_department_id 
  from db_test.org_structure where id = 4
 union
   select os.id, os.department_name, os.parent_department_id 
    from db_test.org_structure os
    join report_structure rs on rs.parent_department_id = os.id
 )
 select
  *
 from report_structure
 
-- Window Functions

select
   department_id,
   last_name,
   salary,
   rank() over (partition by department_id order by salary desc)
from
   db_test.employees;
----------------------------------------------
select
   department_id,
   last_name,
   salary,
   lead(salary,2) over (partition by department_id order by salary desc)
from
   db_test.employees;
----------------------------------------------
select
   department_id,
   last_name,
   salary,
   lag(salary,2) over (partition by department_id order by salary desc)
from
   db_test.employees;
----------------------------------------------
select
  department_id,
  last_name,
  salary,
  round((cume_dist() over (order by salary desc) * 100)::numeric, 2)
from 
  db_test.employees
order by
   salary desc
----------------------------------------------
select
   department_id,
   last_name,
   salary,
   first_value(salary) over (partition by department_id order by salary asc) first_sal
from
   db_test.employees;
----------------------------------------------   
select
   department_id,
   salary,
   avg(salary) over (partition by department_id)
from
   db_test.employees;
----------------------------------------------   
select
   department_id,
   salary,
   ntile(4) over (partition by department_id order by salary desc) as quartile
from
   db_test.employees;
----------------------------------------------
select
   department_id,
   last_name,
   salary,
   nth_value(salary,5) OVER (partition by department_id order by salary desc)
from
   db_test.employees;
   
-- Subqueries & Joins

select
   e1.last_name,
   e1.salary,
   e1.department_id,
   (select avg(salary) from  db_test.employees e2 where e1.department_id = e2.department_id)
from
   db_test.employees e1
   
----------------------------------------------
select
   cd.department_name, count(*)
from
   db_test.employees e
join
   db_test.company_departments cd
on
   e.department_id = cd.id
group by
   cd.department_name
order by 
    2 desc
----------------------------------------------	
select
   round(avg(e1.salary), 2)
from
   (select * from db_test.employees where salary > 100000) e1
   
-- ROLLUP & CUBE
   
select
   department_id
from
   db_test.employees e1
where
   (select max(salary) from db_test.employees e2) = e1.salary
   
select
   cr.country_name, cr.region_name, count(e.*)
from
   db_test.employees e
join
   db_test.company_regions cr
on
   e.region_id = cr.id
group by
   rollup(cr.country_name, cr.region_name)
order by
   cr.country_name, cr.region_name
------------------------------------------------------
select
   cr.country_name,
   cr.region_name,
   cd.department_name,
   count(e.*)
from
  db_test.employees e
join
   db_test.company_regions cr
on
   e.region_id = cr.id
join
   db_test.company_departments cd
on
   e.department_id = cd.id
group by 
   cube(cr.country_name,
        cr.region_name,
        cd.department_name)
order by
   cr.country_name,
   cr.region_name,
   cd.department_name
 
--- Aggregate & Simple Functions

select
  upper(department_name)
from
  db_test.company_departments
------------------------------------------------------
select
  initcap(department_name)
from
  db_test.company_departments
------------------------------------------------------
select
  lower(department_name)
from
  db_test.company_departments
------------------------------------------------------
select
   substring(‘abcdefghijk’ from 1 for 3) test_string

------------------------------------------------------
select distinct
  job_titles
from
  db_test.employees
where
  job_title similar to ‘(vp%|web%)’
------------------------------------------------------  
select distinct
  job_titles
from
  db_test.employees
where
  job_title similar to ‘vp (a|m)%’
------------------------------------------------------  
select
   ceil(avg(salary))
from
   db_test.employees
------------------------------------------------------   
select
   trunc( avg(salary),2)
from
    db_test.employees
	
--- Mean, Variance, Standard Deviation

select
  department_id, 
  sum(salary), 
  round(avg(salary),2), 
  round(var_pop(salary),2), 
  round(stddev_pop(salary),2)
from 
  db_test.employees
group by department_id


--- Data Cleaning

-- Standardize Date Format
/* select TO_CHAR(date_join,'dd:mm:yyyy') date_of_joining
From db_test.employees */

update db_test.employees
set date_join = TO_CHAR(date_join,'dd:mm:yyyy')
-----------------------------------------------

/* select a.department_id, a.last_name, b.department_id, b.last_name, ISNULL(a.last_name,b.last_name)
from   db_test.org_structure a
join   db_test.org_structure b
	on a.department_id = b.department_id
	and a.[region_id ] <> b.[region_id ]
where a.last_name is null */

update a
set last_name = ISNULL(a.last_name,b.last_name)
from db_test.org_structure a
join db_test.org_structure b
	on a.department_id = b.department_id
	and a.[region_id ] <> b.[region_id ]
where a.last_name is null

-----------------------------------------------

/*select region_nrth_amr
, CASE When region_nrth_amr = 'Y' THEN 'Yes'
	   When region_nrth_amr = 'N' THEN 'No'
	   ELSE region_nrth_amr
	   END
from db_test.org_structure*/


update db_test.org_structure
set region_nrth_amr = CASE When region_nrth_amr = 'Y' THEN 'Yes'
						   When region_nrth_amr = 'N' THEN 'No'
						   ELSE region_nrth_amr
					  END
					  
-------------------------------------------------
/*select
SUBSTRING(country_name, 1, POSITION(',' in country_name) -1 ) as country
, SUBSTRING(country_name, POSITION(',' in country_name) + 1 , LEN(country_name)) as country
from db_test.company_regions */

update db_test.company_regions
set region_name = SUBSTRING(country_name, 1, POSITION(',' in country_name) -1 )

update db_test.company_regions
set cntry_name = SUBSTRING(country_name, POSITION(',' in country_name) + 1 , LEN(country_name))