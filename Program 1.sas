/* Generated Code (IMPORT) */
/* Source File: fraud_detection.csv */
/* Source Path: /home/u63751674/fraud_detection */
/* Code generated on: 6/10/24, 10:48 PM */

%web_drop_table(WORK.IMPORT);


FILENAME REFFILE '/home/u63751674/fraud_detection/fraud_detection.csv';

PROC IMPORT DATAFILE=REFFILE
	DBMS=CSV
	OUT=WORK.IMPORT;
	GETNAMES=YES;
RUN;

PROC CONTENTS DATA=WORK.IMPORT; RUN;


%web_open_table(WORK.IMPORT);

PROC CONTENTS DATA=WORK.IMPORT; 
RUN;

/* Step 2: Basic data cleaning - removing missing values */
data transactions_clean;
    set WORK.IMPORT;
    /* Check and remove rows with any missing values */
    if cmiss(of _all_) then delete;
run;

/* Verify the cleaned data */
proc print data=transactions_clean(obs=5); 
run;

/* Step 3: Calculate mean and standard deviation of transaction amounts */
proc means data=transactions_clean mean std;
    var amount;
    output out=stats mean=mean_amount std=std_amount;
run;

/* Step 4: Merge the statistics back into the original dataset */
data transactions_z;
    if _N_ = 1 then set stats;
    set transactions_clean;
    z_score = (amount - mean_amount) / std_amount;
run;

/* Step 5: Identify transactions with z-score > 3 or < -3 */
data transactions_z_outliers;
    set transactions_z;
    if abs(z_score) > 3;
run;

/* Step 6: Display outliers */
proc print data=transactions_z_outliers;
run;

