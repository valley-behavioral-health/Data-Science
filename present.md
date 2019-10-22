# Exploration of Valley Behavioral Health dataset.

The initial 5 row sample we received contained 234 columns. This contained demographic data, the partner program the 
client utilizes, and a number of basic, clinical test results. All were grouped by date into blocks of 90 days. We 
requested more granular data, daily if possible, and that our final dataset include any and all clinical data available 
in addition to what was originally given. 

We planned to use daily data for each patient as sequences to be consumed by an LSTM or similar model, perhaps 
incorporating reinforcement learning as well. Automatic daily updates with new clinical data would feed into our 
deployed model to give updated predictions. Ideally, we would be able to incorporate our model into the partner's 
existing electronic health record (EHR) system. Clinicians would otherwise need to input data into our system as well 
as the EHR.

We were also interested in including clinician notes as our research suggested that many risk factors for 
hospitalization, like impending job loss, homelessness, and other socioeconomic factors, are rarely included in 
tabular data, but often appear in dictated or writen notes. Notes would be subject to natural language processing for 
topic modeling and added to other sequential, daily data. 

Unfortunately our partners were unable to successful anonymize the notes and did not engage with our offers to help 
them do so. In addition we did not receive more than our sample data for over 4 weeks.

In the end we received 15 million rows of daily data with the same columns we received in our sample. Below we 
describe our process of data exploration in a Jupyter Notebook.

Upgrade libraries.

```python
# !pip install dask --upgrade
# !pip install dask[complete]
# !pip install 'fsspec>=0.3.3'
# !pip install s3fs --upgrade
# !pip install msgpack==0.5.6
# !pip install tqdm
```

Import libraries.
```python
import boto3
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client
from tqdm import tqdm
import pandas as pd
from sagemaker import get_execution_role
from d_types import dtypes
import s3fs
```

Confirm Dask client.

```python
client = Client()
client
```

<table style="border: 2px solid white;">
<tr>
<td style="vertical-align: top; border: 0px solid white">
<h3 style="text-align: left;">Client</h3>
<ul style="text-align: left; list-style: none; margin: 0; padding: 0;">
  <li><b>Scheduler: </b>tcp://127.0.0.1:44433</li>
  <li><b>Dashboard: </b><a href='http://127.0.0.1:36537/status' target='_blank'>http://127.0.0.1:36537/status</a>
</ul>
</td>
<td style="vertical-align: top; border: 0px solid white">
<h3 style="text-align: left;">Cluster</h3>
<ul style="text-align: left; list-style:none; margin: 0; padding: 0;">
  <li><b>Workers: </b>4</li>
  <li><b>Cores: </b>4</li>
  <li><b>Memory: </b>16.82 GB</li>
</ul>
</td>
</tr>
</table>



```python
pd.set_option('display.max_columns', 250)
pd.set_option('display.max_rows', 250)
```

Import to Dask DataFrame.

```python
role = get_execution_role()

bucket='vbhdata'
data_key = 'VBH_hospdata.txt'
data_location = 's3://{}/{}'.format(bucket, data_key)

```


```python
df = dd.read_csv(data_location,
                 sep='|',
                 dtype=dtypes,
                 blocksize=64000000,
                 low_memory=False)
```

Confirm import.

```python
df.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>_Date</th>
      <th>Clientid</th>
      <th>ClientStatus</th>
      <th>IsMale</th>
      <th>DOB</th>
      <th>PrimaryProgramName</th>
      <th>MilitaryStatus</th>
      <th>Age</th>
      <th>HispanicOrigin</th>
      <th>Race</th>
      <th>Ethnicity</th>
      <th>MaritalStatus</th>
      <th>CurrentlyHomeless</th>
      <th>LivingArrangement</th>
      <th>ClientState</th>
      <th>Unnamed: 15</th>
      <th>ClientCounty</th>
      <th>PrimaryLanguage</th>
      <th>EducationLevel</th>
      <th>EducationStatus</th>
      <th>FinanciallyResponsible</th>
      <th>EmploymentStatus</th>
      <th>EmploymentInformation</th>
      <th>ClientMonthlyIncome</th>
      <th>ClientAnnualIncome</th>
      <th>HouseholdAnnualIncome</th>
      <th>PrimarySource</th>
      <th>SmokingStatus</th>
      <th>AgeOfFirstTobaccoUse</th>
      <th>HeadOfHousehold</th>
      <th>NumberInHousehold</th>
      <th>NumberofDependents</th>
      <th>ForensicTreatment</th>
      <th>JusticeSystemInvolvement</th>
      <th>Admissions</th>
      <th>Crisis</th>
      <th>SelfHarm</th>
      <th>HarmToOthers</th>
      <th>HarmToProperty</th>
      <th>Pgm_A &amp; D Adult Outpatient</th>
      <th>Pgm_A &amp; D Youth Day Tx</th>
      <th>Pgm_A.T.I.- J.D.O.T.</th>
      <th>Pgm_A.T.I.-C.O.R.E. Outpatient</th>
      <th>Pgm_A.T.I.-C.O.R.E. Residential</th>
      <th>Pgm_Acute Chldrns Extended Sv</th>
      <th>Pgm_Adult Autism Center of Lifetime Learning</th>
      <th>Pgm_Adult Centralized Eval</th>
      <th>Pgm_Alliance House Apts</th>
      <th>Pgm_Assertive Outreach Team</th>
      <th>Pgm_Beacon Heights Elem Sch</th>
      <th>Pgm_C.O.R.E.2 Outpatient</th>
      <th>Pgm_C.O.R.E.2 Residential</th>
      <th>Pgm_Carmen B Pingree Elementary School</th>
      <th>Pgm_Carmen B Pingree In Home</th>
      <th>Pgm_Carmen B Pingree Preschool</th>
      <th>Pgm_Centralized Intake Clinic</th>
      <th>Pgm_Childrens Outpatient Srvc</th>
      <th>Pgm_Childrens Outpatient West</th>
      <th>Pgm_Community Based Prov Ntwk</th>
      <th>Pgm_Contract-Granite School District</th>
      <th>Pgm_Contract-Jordan School District</th>
      <th>Pgm_D.B.T. Day Tx Program</th>
      <th>Pgm_Federal Forensics Grant</th>
      <th>Pgm_Flexcare New Choices</th>
      <th>Pgm_Forensic Unit</th>
      <th>Pgm_Fresh Start/Recovery Prog</th>
      <th>Pgm_H.S.S.C Am Fork</th>
      <th>Pgm_H.S.S.C Highland</th>
      <th>Pgm_H.S.S.C Riverton</th>
      <th>Pgm_H.S.S.C. Boise</th>
      <th>Pgm_H.S.S.C. Lab Services</th>
      <th>Pgm_H.S.S.C. Layton</th>
      <th>Pgm_H.S.S.C. Orem</th>
      <th>Pgm_Homefront</th>
      <th>Pgm_Independent Living</th>
      <th>Pgm_Inpatient Hospital Summit Adult</th>
      <th>Pgm_Inpatient Hospital Tooele Adult</th>
      <th>Pgm_Inpatient Hospital Tooele Youth</th>
      <th>Pgm_K.I.D.S</th>
      <th>Pgm_Lab Services</th>
      <th>Pgm_Masters Program</th>
      <th>Pgm_Medical Records</th>
      <th>Pgm_No Primary program</th>
      <th>Pgm_North Valley Adult Pgm</th>
      <th>Pgm_Pheasant Hollow</th>
      <th>Pgm_Residential Tooele Youth</th>
      <th>Pgm_Robert Frost Elem Sch</th>
      <th>Pgm_S.R.S.</th>
      <th>Pgm_Safe Haven 1</th>
      <th>Pgm_Safe Haven 2</th>
      <th>Pgm_School Based Program</th>
      <th>Pgm_Summit County A &amp; D</th>
      <th>Pgm_Summit County Mental Hlth</th>
      <th>Pgm_Summit Jail</th>
      <th>Pgm_Summit JRI Jail</th>
      <th>Pgm_Summit JRI Outpt</th>
      <th>Pgm_Summit Prevention</th>
      <th>Pgm_Summit School Contractor</th>
      <th>Pgm_TMS</th>
      <th>Pgm_Tooele After School Program</th>
      <th>Pgm_Tooele County A &amp; D</th>
      <th>Pgm_Tooele County M. H.</th>
      <th>Pgm_Tooele County Prevention</th>
      <th>Pgm_Tooele Food Bank</th>
      <th>Pgm_Tooele I-Wrap</th>
      <th>Pgm_Tooele Jail</th>
      <th>Pgm_Tooele JRI Jail</th>
      <th>Pgm_Tooele JRI Outpt</th>
      <th>Pgm_Tooele Resource Center</th>
      <th>Pgm_Tooele Wendover</th>
      <th>Pgm_Tooele Youth Services</th>
      <th>Pgm_USH Summit Youth</th>
      <th>Pgm_USH Tooele Adult</th>
      <th>Pgm_Valley EPIC Bldg A</th>
      <th>Pgm_Valley EPIC Bldg B</th>
      <th>Pgm_Valley EPIC Bldg C</th>
      <th>Pgm_Valley EPIC Bldg D</th>
      <th>Pgm_Valley EPIC Outpatient Recovery Mngmnt</th>
      <th>Pgm_Valley Plaza</th>
      <th>Pgm_Valley Storefront</th>
      <th>Pgm_Valley Woods</th>
      <th>Pgm_ValleyAIM</th>
      <th>Pgm_ValleyPhoenix</th>
      <th>Pgm_ValleyWest</th>
      <th>PosTest_6-Acetylmorphine</th>
      <th>PosTest_7-Aminoclonazepam</th>
      <th>PosTest_7-Hydroxyquetiapine</th>
      <th>PosTest_9-hydroxyrisperidone</th>
      <th>PosTest_Albumin</th>
      <th>PosTest_Alk Phos</th>
      <th>PosTest_Alprazolam</th>
      <th>PosTest_ALT</th>
      <th>PosTest_Amphetamine</th>
      <th>PosTest_Amphetamines</th>
      <th>PosTest_Anion Gap</th>
      <th>PosTest_aOH-Alprazolam</th>
      <th>PosTest_Aripiprazole</th>
      <th>PosTest_AST</th>
      <th>PosTest_Barbiturates</th>
      <th>PosTest_Basophil#</th>
      <th>PosTest_Basophil%</th>
      <th>PosTest_Benzodiazepines</th>
      <th>PosTest_Benzoylecgonine</th>
      <th>PosTest_BUN</th>
      <th>PosTest_Buprenorphine</th>
      <th>PosTest_Bupropion</th>
      <th>PosTest_Calcium</th>
      <th>PosTest_Carisoprodol</th>
      <th>PosTest_Chloride</th>
      <th>PosTest_Cholesterol</th>
      <th>PosTest_Clozapine</th>
      <th>PosTest_Codeine</th>
      <th>PosTest_Desmethylolanzapine</th>
      <th>PosTest_Diazyme Potassium</th>
      <th>PosTest_Diazyme Sodium</th>
      <th>PosTest_EDDP</th>
      <th>PosTest_EGFR (African American)</th>
      <th>PosTest_EGFR (Non-African American)</th>
      <th>PosTest_Eosinophil#</th>
      <th>PosTest_Eosinophil%</th>
      <th>PosTest_EtG</th>
      <th>PosTest_Ethyl Alcohol (mg/dL)</th>
      <th>PosTest_EtS</th>
      <th>PosTest_Fentanyl</th>
      <th>PosTest_Gabapentin</th>
      <th>PosTest_Glucose</th>
      <th>PosTest_Haloperidol</th>
      <th>PosTest_HbA1c</th>
      <th>PosTest_HCT</th>
      <th>PosTest_HDL</th>
      <th>PosTest_HGB</th>
      <th>PosTest_Hydrocodone</th>
      <th>PosTest_Hydromorphone</th>
      <th>PosTest_Hydroxybupropion</th>
      <th>PosTest_LDL Direct</th>
      <th>PosTest_LDL/HDL Ratio</th>
      <th>PosTest_Lorazepam</th>
      <th>PosTest_Lymphocyte#</th>
      <th>PosTest_Lymphocyte%</th>
      <th>PosTest_MCH</th>
      <th>PosTest_MCHC</th>
      <th>PosTest_MCV</th>
      <th>PosTest_MDA</th>
      <th>PosTest_MDEA</th>
      <th>PosTest_MDMA</th>
      <th>PosTest_Meprobamate</th>
      <th>PosTest_Methadone</th>
      <th>PosTest_Methamphetamine</th>
      <th>PosTest_Mitragynine</th>
      <th>PosTest_Monocyte#</th>
      <th>PosTest_Monocyte%</th>
      <th>PosTest_Morphine</th>
      <th>PosTest_MPV</th>
      <th>PosTest_N-Desmethyclozapine</th>
      <th>PosTest_Neutrophil#</th>
      <th>PosTest_Neutrophil%</th>
      <th>PosTest_Norbuprenorphine</th>
      <th>PosTest_Nordiazepam</th>
      <th>PosTest_Norfentanyl</th>
      <th>PosTest_Norhydrocodone</th>
      <th>PosTest_Noroxycodone</th>
      <th>PosTest_O-desmethyltramadol</th>
      <th>PosTest_Olanzapine</th>
      <th>PosTest_Oxazepam</th>
      <th>PosTest_Oxycodone</th>
      <th>PosTest_Oxymorphone</th>
      <th>PosTest_PLT</th>
      <th>PosTest_Potassium</th>
      <th>PosTest_Pregabalin</th>
      <th>PosTest_Protein</th>
      <th>PosTest_Quetiapine</th>
      <th>PosTest_RBC</th>
      <th>PosTest_RDW-CV</th>
      <th>PosTest_Risperidone</th>
      <th>PosTest_Ritalinic Acid</th>
      <th>PosTest_Temazepam</th>
      <th>PosTest_THC-COOH</th>
      <th>PosTest_Total Bilirubin</th>
      <th>PosTest_Total CO2</th>
      <th>PosTest_Tramadol</th>
      <th>PosTest_Triglycerides</th>
      <th>PosTest_TSH</th>
      <th>PosTest_UR-144 5-Pentanoic acid metabolite</th>
      <th>PosTest_VLDL</th>
      <th>PosTest_WBC</th>
      <th>PosTest_Ziprasidone</th>
      <th>Diag_Anxiety, dissociative, stress-related, somatoform and other nonpsychotic mental disorders</th>
      <th>Diag_Behavioral and emotional disorders with onset usually occurring in childhood and adolescence</th>
      <th>Diag_Behavioral syndromes associated with physiological disturbances and physical factors</th>
      <th>Diag_Disorders of adult personality and behavior</th>
      <th>Diag_Intellectual disabilities</th>
      <th>Diag_Mental and behavioral disorders due to psychoactive substance use</th>
      <th>Diag_Mental disorders due to known physiological conditions</th>
      <th>Diag_Mood (affective) disorders</th>
      <th>Diag_No Diagnosis</th>
      <th>Diag_Others</th>
      <th>Diag_Pervasive and specific developmental disorders</th>
      <th>Diag_Schizophrenia, schizotypal, delusional, and other non-mood psychotic disorders</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-01-01</td>
      <td>940.0</td>
      <td>N</td>
      <td>1.0</td>
      <td>1951-04-22</td>
      <td>No Primary program</td>
      <td>No</td>
      <td>68.0</td>
      <td>Not of Hispanic Origin</td>
      <td>White</td>
      <td>Not of Hispanic Origin</td>
      <td>Single</td>
      <td>NaN</td>
      <td>Institutional Setting</td>
      <td>UT</td>
      <td>841XX</td>
      <td>Salt Lake                                     ...</td>
      <td>English</td>
      <td>12th Grade</td>
      <td>Not currently enrolled</td>
      <td>Y</td>
      <td>Disabled - Not in Labor Force</td>
      <td>NaN</td>
      <td>431.0</td>
      <td>5172.0</td>
      <td>5172.0</td>
      <td>Legal Employment, Wages and Salary</td>
      <td>CURRENT EVERDAY SMOKER/E-CIG USER</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>Not applicable</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-01-01</td>
      <td>1060.0</td>
      <td>N</td>
      <td>0.0</td>
      <td>1947-11-21</td>
      <td>No Primary program</td>
      <td>No</td>
      <td>71.0</td>
      <td>Not of Hispanic Origin</td>
      <td>White</td>
      <td>Not of Hispanic Origin</td>
      <td>Separated</td>
      <td>NaN</td>
      <td>Institutional Setting</td>
      <td>UT</td>
      <td>840XX</td>
      <td>Salt Lake                                     ...</td>
      <td>English</td>
      <td>7th Grade</td>
      <td>Not currently enrolled</td>
      <td>Y</td>
      <td>Disabled - Not in Labor Force</td>
      <td>NaN</td>
      <td>700.0</td>
      <td>8400.0</td>
      <td>8400.0</td>
      <td>Legal Employment, Wages and Salary</td>
      <td>NEVER SMOKED/VAPED</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Not applicable</td>
      <td>Not applicable</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-01-01</td>
      <td>1250.0</td>
      <td>N</td>
      <td>0.0</td>
      <td>1959-12-07</td>
      <td>No Primary program</td>
      <td>No</td>
      <td>59.0</td>
      <td>Not of Hispanic Origin</td>
      <td>Pacific Islander or Native Hawaiian/White</td>
      <td>Not of Hispanic Origin</td>
      <td>Married</td>
      <td>Not Homeless</td>
      <td>24-hour Residential Care</td>
      <td>UT</td>
      <td>841XX</td>
      <td>Salt Lake                                     ...</td>
      <td>English</td>
      <td>12th Grade</td>
      <td>Not currently enrolled</td>
      <td>Y</td>
      <td>Disabled - Not in Labor Force</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Pension, Retirement Benefits, Social Security</td>
      <td>CURRENT SOME DAY SMOKER/E-CIG USER</td>
      <td>0.0</td>
      <td>Y</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Not applicable</td>
      <td>Not applicable</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-01-01</td>
      <td>1831.0</td>
      <td>N</td>
      <td>0.0</td>
      <td>1946-08-24</td>
      <td>No Primary program</td>
      <td>No</td>
      <td>73.0</td>
      <td>Not of Hispanic Origin</td>
      <td>White</td>
      <td>Not of Hispanic Origin</td>
      <td>Married</td>
      <td>NaN</td>
      <td>Private Residence-Independent</td>
      <td>UT</td>
      <td>840XX</td>
      <td>Salt Lake                                     ...</td>
      <td>English</td>
      <td>15</td>
      <td>Not currently enrolled</td>
      <td>Y</td>
      <td>Homemaker</td>
      <td>NaN</td>
      <td>5000.0</td>
      <td>60000.0</td>
      <td>60000.0</td>
      <td>Pension, Retirement Benefits, Social Security</td>
      <td>NOT APPLICABLE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-01-01</td>
      <td>2541.0</td>
      <td>N</td>
      <td>1.0</td>
      <td>1971-08-15</td>
      <td>No Primary program</td>
      <td>No</td>
      <td>48.0</td>
      <td>Mexican</td>
      <td>Other single race</td>
      <td>Mexican</td>
      <td>Single</td>
      <td>NaN</td>
      <td>Private Residence-Independent</td>
      <td>UT</td>
      <td>841XX</td>
      <td>Salt Lake                                     ...</td>
      <td>English</td>
      <td>11th Grade</td>
      <td>Not currently enrolled</td>
      <td>Y</td>
      <td>Unemployed, Not Seeking Work</td>
      <td>NaN</td>
      <td>703.0</td>
      <td>8436.0</td>
      <td>8436.0</td>
      <td>None</td>
      <td>CURRENT EVERDAY SMOKER/E-CIG USER</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Not applicable</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

Check for missing values and take a look at what information we have.


```python
df2 = df.isna().sum()
```


```python
df2.compute()
```
    _Date                                                                                                       0
    Clientid                                                                                                    1
    ClientStatus                                                                                                1
    IsMale                                                                                                      1
    DOB                                                                                                      6604
    PrimaryProgramName                                                                                          1
    MilitaryStatus                                                                                        5522299
    Age                                                                                                      6604
    HispanicOrigin                                                                                        5975616
    Race                                                                                                  3054512
    Ethnicity                                                                                             7739755
    MaritalStatus                                                                                               1
    CurrentlyHomeless                                                                                     9196951
    LivingArrangement                                                                                     4483892
    ClientState                                                                                            713453
    Unnamed: 15                                                                                            724346
    ClientCounty                                                                                          4755888
    PrimaryLanguage                                                                                       4981986
    EducationLevel                                                                                        6285637
    EducationStatus                                                                                       5744903
    FinanciallyResponsible                                                                                      1
    EmploymentStatus                                                                                      6127128
    EmploymentInformation                                                                                14929955
    ClientMonthlyIncome                                                                                   6359168
    ClientAnnualIncome                                                                                    6372118
    HouseholdAnnualIncome                                                                                 7437706
    PrimarySource                                                                                         6550880
    SmokingStatus                                                                                         6292572
    AgeOfFirstTobaccoUse                                                                                  7709215
    HeadOfHousehold                                                                                      13357160
    NumberInHousehold                                                                                     6292092
    NumberofDependents                                                                                    5909568
    ForensicTreatment                                                                                     9037521
    JusticeSystemInvolvement                                                                             10595422
    Admissions                                                                                                  1
    Crisis                                                                                                      1
    SelfHarm                                                                                                    1
    HarmToOthers                                                                                                1
    HarmToProperty                                                                                              1
    Pgm_A & D Adult Outpatient                                                                                  1
    Pgm_A & D Youth Day Tx                                                                                      1
    Pgm_A.T.I.- J.D.O.T.                                                                                        1
    Pgm_A.T.I.-C.O.R.E. Outpatient                                                                              1
    Pgm_A.T.I.-C.O.R.E. Residential                                                                             1
    Pgm_Acute Chldrns Extended Sv                                                                               1
    Pgm_Adult Autism Center of Lifetime Learning                                                                1
    Pgm_Adult Centralized Eval                                                                                  1
    Pgm_Alliance House Apts                                                                                     1
    Pgm_Assertive Outreach Team                                                                                 1
    Pgm_Beacon Heights Elem Sch                                                                                 1
    Pgm_C.O.R.E.2 Outpatient                                                                                    1
    Pgm_C.O.R.E.2 Residential                                                                                   1
    Pgm_Carmen B Pingree Elementary School                                                                      1
    Pgm_Carmen B Pingree In Home                                                                                1
    Pgm_Carmen B Pingree Preschool                                                                              1
    Pgm_Centralized Intake Clinic                                                                               1
    Pgm_Childrens Outpatient Srvc                                                                               1
    Pgm_Childrens Outpatient West                                                                               1
    Pgm_Community Based Prov Ntwk                                                                               1
    Pgm_Contract-Granite School District                                                                        1
    Pgm_Contract-Jordan School District                                                                         1
    Pgm_D.B.T. Day Tx Program                                                                                   1
    Pgm_Federal Forensics Grant                                                                                 1
    Pgm_Flexcare New Choices                                                                                    1
    Pgm_Forensic Unit                                                                                           1
    Pgm_Fresh Start/Recovery Prog                                                                               1
    Pgm_H.S.S.C Am Fork                                                                                         1
    Pgm_H.S.S.C Highland                                                                                        1
    Pgm_H.S.S.C Riverton                                                                                        1
    Pgm_H.S.S.C. Boise                                                                                          1
    Pgm_H.S.S.C. Lab Services                                                                                   1
    Pgm_H.S.S.C. Layton                                                                                         1
    Pgm_H.S.S.C. Orem                                                                                           1
    Pgm_Homefront                                                                                               1
    Pgm_Independent Living                                                                                      1
    Pgm_Inpatient Hospital Summit Adult                                                                         1
    Pgm_Inpatient Hospital Tooele Adult                                                                         1
    Pgm_Inpatient Hospital Tooele Youth                                                                         1
    Pgm_K.I.D.S                                                                                                 1
    Pgm_Lab Services                                                                                            1
    Pgm_Masters Program                                                                                         1
    Pgm_Medical Records                                                                                         1
    Pgm_No Primary program                                                                                      1
    Pgm_North Valley Adult Pgm                                                                                  1
    Pgm_Pheasant Hollow                                                                                         1
    Pgm_Residential Tooele Youth                                                                                1
    Pgm_Robert Frost Elem Sch                                                                                   1
    Pgm_S.R.S.                                                                                                  1
    Pgm_Safe Haven 1                                                                                            1
    Pgm_Safe Haven 2                                                                                            1
    Pgm_School Based Program                                                                                    1
    Pgm_Summit County A & D                                                                                     1
    Pgm_Summit County Mental Hlth                                                                               1
    Pgm_Summit Jail                                                                                             1
    Pgm_Summit JRI Jail                                                                                         1
    Pgm_Summit JRI Outpt                                                                                        1
    Pgm_Summit Prevention                                                                                       1
    Pgm_Summit School Contractor                                                                                1
    Pgm_TMS                                                                                                     1
    Pgm_Tooele After School Program                                                                             1
    Pgm_Tooele County A & D                                                                                     1
    Pgm_Tooele County M. H.                                                                                     1
    Pgm_Tooele County Prevention                                                                                1
    Pgm_Tooele Food Bank                                                                                        1
    Pgm_Tooele I-Wrap                                                                                           1
    Pgm_Tooele Jail                                                                                             1
    Pgm_Tooele JRI Jail                                                                                         1
    Pgm_Tooele JRI Outpt                                                                                        1
    Pgm_Tooele Resource Center                                                                                  1
    Pgm_Tooele Wendover                                                                                         1
    Pgm_Tooele Youth Services                                                                                   1
    Pgm_USH Summit Youth                                                                                        1
    Pgm_USH Tooele Adult                                                                                        1
    Pgm_Valley EPIC Bldg A                                                                                      1
    Pgm_Valley EPIC Bldg B                                                                                      1
    Pgm_Valley EPIC Bldg C                                                                                      1
    Pgm_Valley EPIC Bldg D                                                                                      1
    Pgm_Valley EPIC Outpatient Recovery Mngmnt                                                                  1
    Pgm_Valley Plaza                                                                                            1
    Pgm_Valley Storefront                                                                                       1
    Pgm_Valley Woods                                                                                            1
    Pgm_ValleyAIM                                                                                               1
    Pgm_ValleyPhoenix                                                                                           1
    Pgm_ValleyWest                                                                                              1
    PosTest_6-Acetylmorphine                                                                                    1
    PosTest_7-Aminoclonazepam                                                                                   1
    PosTest_7-Hydroxyquetiapine                                                                                 1
    PosTest_9-hydroxyrisperidone                                                                                1
    PosTest_Albumin                                                                                             1
    PosTest_Alk Phos                                                                                            1
    PosTest_Alprazolam                                                                                          1
    PosTest_ALT                                                                                                 1
    PosTest_Amphetamine                                                                                         1
    PosTest_Amphetamines                                                                                        1
    PosTest_Anion Gap                                                                                           1
    PosTest_aOH-Alprazolam                                                                                      1
    PosTest_Aripiprazole                                                                                        1
    PosTest_AST                                                                                                 1
    PosTest_Barbiturates                                                                                        1
    PosTest_Basophil#                                                                                           1
    PosTest_Basophil%                                                                                           1
    PosTest_Benzodiazepines                                                                                     1
    PosTest_Benzoylecgonine                                                                                     1
    PosTest_BUN                                                                                                 1
    PosTest_Buprenorphine                                                                                       1
    PosTest_Bupropion                                                                                           1
    PosTest_Calcium                                                                                             1
    PosTest_Carisoprodol                                                                                        1
    PosTest_Chloride                                                                                            1
    PosTest_Cholesterol                                                                                         1
    PosTest_Clozapine                                                                                           1
    PosTest_Codeine                                                                                             1
    PosTest_Desmethylolanzapine                                                                                 1
    PosTest_Diazyme Potassium                                                                                   1
    PosTest_Diazyme Sodium                                                                                      1
    PosTest_EDDP                                                                                                1
    PosTest_EGFR (African American)                                                                             1
    PosTest_EGFR (Non-African American)                                                                         1
    PosTest_Eosinophil#                                                                                         1
    PosTest_Eosinophil%                                                                                         1
    PosTest_EtG                                                                                                 1
    PosTest_Ethyl Alcohol (mg/dL)                                                                               1
    PosTest_EtS                                                                                                 1
    PosTest_Fentanyl                                                                                            1
    PosTest_Gabapentin                                                                                          1
    PosTest_Glucose                                                                                             1
    PosTest_Haloperidol                                                                                         1
    PosTest_HbA1c                                                                                               1
    PosTest_HCT                                                                                                 1
    PosTest_HDL                                                                                                 1
    PosTest_HGB                                                                                                 1
    PosTest_Hydrocodone                                                                                         1
    PosTest_Hydromorphone                                                                                       1
    PosTest_Hydroxybupropion                                                                                    1
    PosTest_LDL Direct                                                                                          1
    PosTest_LDL/HDL Ratio                                                                                       1
    PosTest_Lorazepam                                                                                           1
    PosTest_Lymphocyte#                                                                                         1
    PosTest_Lymphocyte%                                                                                         1
    PosTest_MCH                                                                                                 1
    PosTest_MCHC                                                                                                1
    PosTest_MCV                                                                                                 1
    PosTest_MDA                                                                                                 1
    PosTest_MDEA                                                                                                1
    PosTest_MDMA                                                                                                1
    PosTest_Meprobamate                                                                                         1
    PosTest_Methadone                                                                                           1
    PosTest_Methamphetamine                                                                                     1
    PosTest_Mitragynine                                                                                         1
    PosTest_Monocyte#                                                                                           1
    PosTest_Monocyte%                                                                                           1
    PosTest_Morphine                                                                                            1
    PosTest_MPV                                                                                                 1
    PosTest_N-Desmethyclozapine                                                                                 1
    PosTest_Neutrophil#                                                                                         1
    PosTest_Neutrophil%                                                                                         1
    PosTest_Norbuprenorphine                                                                                    1
    PosTest_Nordiazepam                                                                                         1
    PosTest_Norfentanyl                                                                                         1
    PosTest_Norhydrocodone                                                                                      1
    PosTest_Noroxycodone                                                                                        1
    PosTest_O-desmethyltramadol                                                                                 1
    PosTest_Olanzapine                                                                                          1
    PosTest_Oxazepam                                                                                            1
    PosTest_Oxycodone                                                                                           1
    PosTest_Oxymorphone                                                                                         1
    PosTest_PLT                                                                                                 1
    PosTest_Potassium                                                                                           1
    PosTest_Pregabalin                                                                                          1
    PosTest_Protein                                                                                             1
    PosTest_Quetiapine                                                                                          1
    PosTest_RBC                                                                                                 1
    PosTest_RDW-CV                                                                                              1
    PosTest_Risperidone                                                                                         1
    PosTest_Ritalinic Acid                                                                                      1
    PosTest_Temazepam                                                                                           1
    PosTest_THC-COOH                                                                                            1
    PosTest_Total Bilirubin                                                                                     1
    PosTest_Total CO2                                                                                           1
    PosTest_Tramadol                                                                                            1
    PosTest_Triglycerides                                                                                       1
    PosTest_TSH                                                                                                 1
    PosTest_UR-144 5-Pentanoic acid metabolite                                                                  1
    PosTest_VLDL                                                                                                1
    PosTest_WBC                                                                                                 1
    PosTest_Ziprasidone                                                                                         1
    Diag_Anxiety, dissociative, stress-related, somatoform and other nonpsychotic mental disorders              1
    Diag_Behavioral and emotional disorders with onset usually occurring in childhood and adolescence           1
    Diag_Behavioral syndromes associated with physiological disturbances and physical factors                   1
    Diag_Disorders of adult personality and behavior                                                            1
    Diag_Intellectual disabilities                                                                              1
    Diag_Mental and behavioral disorders due to psychoactive substance use                                      1
    Diag_Mental disorders due to known physiological conditions                                                 1
    Diag_Mood (affective) disorders                                                                             1
    Diag_No Diagnosis                                                                                           1
    Diag_Others                                                                                                 1
    Diag_Pervasive and specific developmental disorders                                                         1
    Diag_Schizophrenia, schizotypal, delusional, and other non-mood psychotic disorders                         1
    dtype: int64



We have a lot of demographic data, a number of drug tests, a basic metabolic panel, complete blood count, and some 
diagnoses.

We may have enough to create a model that could recommend more in depth screening for a number of conditions from this 
information, by utilizing other, large datasets with known outcomes. 

For the problem at hand, however, I'm concerned that these values do not change enough over short periods of time to 
predict something like an impending hospitalization. We will take a closer look to determine if this is the case.

Find number of unique clients.

```python
uni = df['Clientid'].unique()
len(uni)
```


    70083

A good number of patients.


Find the total number of hospital additions for the whole dataset.

```python
admins = df['Admissions'].value_counts()
admins.compute()
```

    0.0    15023382
    1.0        3538
    Name: Admissions, dtype: int64



This isn't exactly needle-in-a-haystack territory. It's probably doable, but we'll need robust data to do it. As I 
mentioned above, I'm concerned that we don't have the kind of data we need to make a prediction like an impending 
hospitalization.

### Next, we'll take a look at some individual client timelines and try to get a better understanding of the changes that occur.

To make things faster we'll get only what we need to isolate patients who have been hospitalized.

```python
df1 = df[['_Date', 'Clientid', 'Admissions']]
df1 = df1.dropna()
```

```python
df1.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>_Date</th>
      <th>Clientid</th>
      <th>Admissions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-01-01</td>
      <td>940.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-01-01</td>
      <td>1060.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-01-01</td>
      <td>1250.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-01-01</td>
      <td>1831.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-01-01</td>
      <td>2541.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


```python
df1 = client.persist(df1)
df1 = df1.compute()
```

Check for null values. Dask doesn't seem to like them at all.


```python
df2 = df1.isna().sum()
df2
```

    _Date         0
    Clientid      0
    Admissions    0
    dtype: int64


Filter for hospitalized patients.


```python
hosp_df = df1[df1['Admissions'] == 1.0]
hosp_df = hosp_df.reset_index(drop=True)
hosp_df.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>_Date</th>
      <th>Clientid</th>
      <th>Admissions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-01-01</td>
      <td>365810.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-01-01</td>
      <td>2106790.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-01-01</td>
      <td>2107915.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-01-01</td>
      <td>2127622.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-01-01</td>
      <td>2128389.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


Get list of `Clientid`s of hospitalized pts and filter the whole dataset to get a dataframe of only those patients.


```python
hosp_clients = hosp_df['Clientid'].unique()
hosp_clients_df = df[df['Clientid'].isin(hosp_clients)]
hosp_clients_df = client.persist(hosp_clients_df)
hosp_clients_df = hosp_clients_df.compute()
hosp_clients_df.head()
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>_Date</th>
      <th>Clientid</th>
      <th>ClientStatus</th>
      <th>IsMale</th>
      <th>DOB</th>
      <th>PrimaryProgramName</th>
      <th>MilitaryStatus</th>
      <th>Age</th>
      <th>HispanicOrigin</th>
      <th>Race</th>
      <th>Ethnicity</th>
      <th>MaritalStatus</th>
      <th>CurrentlyHomeless</th>
      <th>LivingArrangement</th>
      <th>ClientState</th>
      <th>Unnamed: 15</th>
      <th>ClientCounty</th>
      <th>PrimaryLanguage</th>
      <th>EducationLevel</th>
      <th>EducationStatus</th>
      <th>FinanciallyResponsible</th>
      <th>EmploymentStatus</th>
      <th>EmploymentInformation</th>
      <th>ClientMonthlyIncome</th>
      <th>ClientAnnualIncome</th>
      <th>HouseholdAnnualIncome</th>
      <th>PrimarySource</th>
      <th>SmokingStatus</th>
      <th>AgeOfFirstTobaccoUse</th>
      <th>HeadOfHousehold</th>
      <th>NumberInHousehold</th>
      <th>NumberofDependents</th>
      <th>ForensicTreatment</th>
      <th>JusticeSystemInvolvement</th>
      <th>Admissions</th>
      <th>Crisis</th>
      <th>SelfHarm</th>
      <th>HarmToOthers</th>
      <th>HarmToProperty</th>
      <th>Pgm_A &amp; D Adult Outpatient</th>
      <th>Pgm_A &amp; D Youth Day Tx</th>
      <th>Pgm_A.T.I.- J.D.O.T.</th>
      <th>Pgm_A.T.I.-C.O.R.E. Outpatient</th>
      <th>Pgm_A.T.I.-C.O.R.E. Residential</th>
      <th>Pgm_Acute Chldrns Extended Sv</th>
      <th>Pgm_Adult Autism Center of Lifetime Learning</th>
      <th>Pgm_Adult Centralized Eval</th>
      <th>Pgm_Alliance House Apts</th>
      <th>Pgm_Assertive Outreach Team</th>
      <th>Pgm_Beacon Heights Elem Sch</th>
      <th>Pgm_C.O.R.E.2 Outpatient</th>
      <th>Pgm_C.O.R.E.2 Residential</th>
      <th>Pgm_Carmen B Pingree Elementary School</th>
      <th>Pgm_Carmen B Pingree In Home</th>
      <th>Pgm_Carmen B Pingree Preschool</th>
      <th>Pgm_Centralized Intake Clinic</th>
      <th>Pgm_Childrens Outpatient Srvc</th>
      <th>Pgm_Childrens Outpatient West</th>
      <th>Pgm_Community Based Prov Ntwk</th>
      <th>Pgm_Contract-Granite School District</th>
      <th>Pgm_Contract-Jordan School District</th>
      <th>Pgm_D.B.T. Day Tx Program</th>
      <th>Pgm_Federal Forensics Grant</th>
      <th>Pgm_Flexcare New Choices</th>
      <th>Pgm_Forensic Unit</th>
      <th>Pgm_Fresh Start/Recovery Prog</th>
      <th>Pgm_H.S.S.C Am Fork</th>
      <th>Pgm_H.S.S.C Highland</th>
      <th>Pgm_H.S.S.C Riverton</th>
      <th>Pgm_H.S.S.C. Boise</th>
      <th>Pgm_H.S.S.C. Lab Services</th>
      <th>Pgm_H.S.S.C. Layton</th>
      <th>Pgm_H.S.S.C. Orem</th>
      <th>Pgm_Homefront</th>
      <th>Pgm_Independent Living</th>
      <th>Pgm_Inpatient Hospital Summit Adult</th>
      <th>Pgm_Inpatient Hospital Tooele Adult</th>
      <th>Pgm_Inpatient Hospital Tooele Youth</th>
      <th>Pgm_K.I.D.S</th>
      <th>Pgm_Lab Services</th>
      <th>Pgm_Masters Program</th>
      <th>Pgm_Medical Records</th>
      <th>Pgm_No Primary program</th>
      <th>Pgm_North Valley Adult Pgm</th>
      <th>Pgm_Pheasant Hollow</th>
      <th>Pgm_Residential Tooele Youth</th>
      <th>Pgm_Robert Frost Elem Sch</th>
      <th>Pgm_S.R.S.</th>
      <th>Pgm_Safe Haven 1</th>
      <th>Pgm_Safe Haven 2</th>
      <th>Pgm_School Based Program</th>
      <th>Pgm_Summit County A &amp; D</th>
      <th>Pgm_Summit County Mental Hlth</th>
      <th>Pgm_Summit Jail</th>
      <th>Pgm_Summit JRI Jail</th>
      <th>Pgm_Summit JRI Outpt</th>
      <th>Pgm_Summit Prevention</th>
      <th>Pgm_Summit School Contractor</th>
      <th>Pgm_TMS</th>
      <th>Pgm_Tooele After School Program</th>
      <th>Pgm_Tooele County A &amp; D</th>
      <th>Pgm_Tooele County M. H.</th>
      <th>Pgm_Tooele County Prevention</th>
      <th>Pgm_Tooele Food Bank</th>
      <th>Pgm_Tooele I-Wrap</th>
      <th>Pgm_Tooele Jail</th>
      <th>Pgm_Tooele JRI Jail</th>
      <th>Pgm_Tooele JRI Outpt</th>
      <th>Pgm_Tooele Resource Center</th>
      <th>Pgm_Tooele Wendover</th>
      <th>Pgm_Tooele Youth Services</th>
      <th>Pgm_USH Summit Youth</th>
      <th>Pgm_USH Tooele Adult</th>
      <th>Pgm_Valley EPIC Bldg A</th>
      <th>Pgm_Valley EPIC Bldg B</th>
      <th>Pgm_Valley EPIC Bldg C</th>
      <th>Pgm_Valley EPIC Bldg D</th>
      <th>Pgm_Valley EPIC Outpatient Recovery Mngmnt</th>
      <th>Pgm_Valley Plaza</th>
      <th>Pgm_Valley Storefront</th>
      <th>Pgm_Valley Woods</th>
      <th>Pgm_ValleyAIM</th>
      <th>Pgm_ValleyPhoenix</th>
      <th>Pgm_ValleyWest</th>
      <th>PosTest_6-Acetylmorphine</th>
      <th>PosTest_7-Aminoclonazepam</th>
      <th>PosTest_7-Hydroxyquetiapine</th>
      <th>PosTest_9-hydroxyrisperidone</th>
      <th>PosTest_Albumin</th>
      <th>PosTest_Alk Phos</th>
      <th>PosTest_Alprazolam</th>
      <th>PosTest_ALT</th>
      <th>PosTest_Amphetamine</th>
      <th>PosTest_Amphetamines</th>
      <th>PosTest_Anion Gap</th>
      <th>PosTest_aOH-Alprazolam</th>
      <th>PosTest_Aripiprazole</th>
      <th>PosTest_AST</th>
      <th>PosTest_Barbiturates</th>
      <th>PosTest_Basophil#</th>
      <th>PosTest_Basophil%</th>
      <th>PosTest_Benzodiazepines</th>
      <th>PosTest_Benzoylecgonine</th>
      <th>PosTest_BUN</th>
      <th>PosTest_Buprenorphine</th>
      <th>PosTest_Bupropion</th>
      <th>PosTest_Calcium</th>
      <th>PosTest_Carisoprodol</th>
      <th>PosTest_Chloride</th>
      <th>PosTest_Cholesterol</th>
      <th>PosTest_Clozapine</th>
      <th>PosTest_Codeine</th>
      <th>PosTest_Desmethylolanzapine</th>
      <th>PosTest_Diazyme Potassium</th>
      <th>PosTest_Diazyme Sodium</th>
      <th>PosTest_EDDP</th>
      <th>PosTest_EGFR (African American)</th>
      <th>PosTest_EGFR (Non-African American)</th>
      <th>PosTest_Eosinophil#</th>
      <th>PosTest_Eosinophil%</th>
      <th>PosTest_EtG</th>
      <th>PosTest_Ethyl Alcohol (mg/dL)</th>
      <th>PosTest_EtS</th>
      <th>PosTest_Fentanyl</th>
      <th>PosTest_Gabapentin</th>
      <th>PosTest_Glucose</th>
      <th>PosTest_Haloperidol</th>
      <th>PosTest_HbA1c</th>
      <th>PosTest_HCT</th>
      <th>PosTest_HDL</th>
      <th>PosTest_HGB</th>
      <th>PosTest_Hydrocodone</th>
      <th>PosTest_Hydromorphone</th>
      <th>PosTest_Hydroxybupropion</th>
      <th>PosTest_LDL Direct</th>
      <th>PosTest_LDL/HDL Ratio</th>
      <th>PosTest_Lorazepam</th>
      <th>PosTest_Lymphocyte#</th>
      <th>PosTest_Lymphocyte%</th>
      <th>PosTest_MCH</th>
      <th>PosTest_MCHC</th>
      <th>PosTest_MCV</th>
      <th>PosTest_MDA</th>
      <th>PosTest_MDEA</th>
      <th>PosTest_MDMA</th>
      <th>PosTest_Meprobamate</th>
      <th>PosTest_Methadone</th>
      <th>PosTest_Methamphetamine</th>
      <th>PosTest_Mitragynine</th>
      <th>PosTest_Monocyte#</th>
      <th>PosTest_Monocyte%</th>
      <th>PosTest_Morphine</th>
      <th>PosTest_MPV</th>
      <th>PosTest_N-Desmethyclozapine</th>
      <th>PosTest_Neutrophil#</th>
      <th>PosTest_Neutrophil%</th>
      <th>PosTest_Norbuprenorphine</th>
      <th>PosTest_Nordiazepam</th>
      <th>PosTest_Norfentanyl</th>
      <th>PosTest_Norhydrocodone</th>
      <th>PosTest_Noroxycodone</th>
      <th>PosTest_O-desmethyltramadol</th>
      <th>PosTest_Olanzapine</th>
      <th>PosTest_Oxazepam</th>
      <th>PosTest_Oxycodone</th>
      <th>PosTest_Oxymorphone</th>
      <th>PosTest_PLT</th>
      <th>PosTest_Potassium</th>
      <th>PosTest_Pregabalin</th>
      <th>PosTest_Protein</th>
      <th>PosTest_Quetiapine</th>
      <th>PosTest_RBC</th>
      <th>PosTest_RDW-CV</th>
      <th>PosTest_Risperidone</th>
      <th>PosTest_Ritalinic Acid</th>
      <th>PosTest_Temazepam</th>
      <th>PosTest_THC-COOH</th>
      <th>PosTest_Total Bilirubin</th>
      <th>PosTest_Total CO2</th>
      <th>PosTest_Tramadol</th>
      <th>PosTest_Triglycerides</th>
      <th>PosTest_TSH</th>
      <th>PosTest_UR-144 5-Pentanoic acid metabolite</th>
      <th>PosTest_VLDL</th>
      <th>PosTest_WBC</th>
      <th>PosTest_Ziprasidone</th>
      <th>Diag_Anxiety, dissociative, stress-related, somatoform and other nonpsychotic mental disorders</th>
      <th>Diag_Behavioral and emotional disorders with onset usually occurring in childhood and adolescence</th>
      <th>Diag_Behavioral syndromes associated with physiological disturbances and physical factors</th>
      <th>Diag_Disorders of adult personality and behavior</th>
      <th>Diag_Intellectual disabilities</th>
      <th>Diag_Mental and behavioral disorders due to psychoactive substance use</th>
      <th>Diag_Mental disorders due to known physiological conditions</th>
      <th>Diag_Mood (affective) disorders</th>
      <th>Diag_No Diagnosis</th>
      <th>Diag_Others</th>
      <th>Diag_Pervasive and specific developmental disorders</th>
      <th>Diag_Schizophrenia, schizotypal, delusional, and other non-mood psychotic disorders</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2019-01-01</td>
      <td>1250.0</td>
      <td>N</td>
      <td>0.0</td>
      <td>1959-12-07</td>
      <td>No Primary program</td>
      <td>No</td>
      <td>59.0</td>
      <td>Not of Hispanic Origin</td>
      <td>Pacific Islander or Native Hawaiian/White</td>
      <td>Not of Hispanic Origin</td>
      <td>Married</td>
      <td>Not Homeless</td>
      <td>24-hour Residential Care</td>
      <td>UT</td>
      <td>841XX</td>
      <td>Salt Lake                                     ...</td>
      <td>English</td>
      <td>12th Grade</td>
      <td>Not currently enrolled</td>
      <td>Y</td>
      <td>Disabled - Not in Labor Force</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Pension, Retirement Benefits, Social Security</td>
      <td>CURRENT SOME DAY SMOKER/E-CIG USER</td>
      <td>0.0</td>
      <td>Y</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Not applicable</td>
      <td>Not applicable</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2019-01-01</td>
      <td>6091.0</td>
      <td>Y</td>
      <td>1.0</td>
      <td>1961-11-17</td>
      <td>ValleyWest</td>
      <td>NaN</td>
      <td>57.0</td>
      <td>Not of Hispanic Origin</td>
      <td>White</td>
      <td>Not of Hispanic Origin</td>
      <td>Single</td>
      <td>NaN</td>
      <td>Private Residence-Dependent</td>
      <td>UT</td>
      <td>841XX</td>
      <td>Salt Lake                                     ...</td>
      <td>English</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Y</td>
      <td>Supported/Trans Employment</td>
      <td>NaN</td>
      <td>787.0</td>
      <td>9444.0</td>
      <td>9444.0</td>
      <td>Legal Employment, Wages and Salary</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2019-01-01</td>
      <td>6430.0</td>
      <td>Y</td>
      <td>0.0</td>
      <td>1944-04-21</td>
      <td>Masters Program</td>
      <td>NaN</td>
      <td>75.0</td>
      <td>Not of Hispanic Origin</td>
      <td>White</td>
      <td>Not of Hispanic Origin</td>
      <td>Divorced</td>
      <td>NaN</td>
      <td>Private Residence-Independent</td>
      <td>UT</td>
      <td>841XX</td>
      <td>Salt Lake                                     ...</td>
      <td>English</td>
      <td>13</td>
      <td>Yes currently enrolled</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>946.0</td>
      <td>11352.0</td>
      <td>11352.0</td>
      <td>Unemployment</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2019-01-01</td>
      <td>9111.0</td>
      <td>Y</td>
      <td>0.0</td>
      <td>1967-02-13</td>
      <td>North Valley Adult Pgm</td>
      <td>No</td>
      <td>52.0</td>
      <td>Not of Hispanic Origin</td>
      <td>Other single race/White</td>
      <td>Not of Hispanic Origin</td>
      <td>Divorced</td>
      <td>Chronically Homeless</td>
      <td>24-hour Residential Care</td>
      <td>UT</td>
      <td>841XX</td>
      <td>Salt Lake                                     ...</td>
      <td>English</td>
      <td>14</td>
      <td>Not currently enrolled</td>
      <td>Y</td>
      <td>Disabled - Not in Labor Force</td>
      <td>NaN</td>
      <td>659.0</td>
      <td>7908.0</td>
      <td>7764.0</td>
      <td>Pension, Retirement Benefits, Social Security</td>
      <td>CURRENT EVERDAY SMOKER/E-CIG USER</td>
      <td>0.0</td>
      <td>Y</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>New-(Justice Involved) OLD-Criminal Court Orde...</td>
      <td>Other</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>2019-01-01</td>
      <td>13540.0</td>
      <td>Y</td>
      <td>0.0</td>
      <td>1948-04-11</td>
      <td>S.R.S.</td>
      <td>No</td>
      <td>71.0</td>
      <td>Not of Hispanic Origin</td>
      <td>White</td>
      <td>Not of Hispanic Origin</td>
      <td>Married</td>
      <td>NaN</td>
      <td>Private Residence-Independent</td>
      <td>UT</td>
      <td>841XX</td>
      <td>NaN</td>
      <td>English</td>
      <td>14</td>
      <td>Yes currently enrolled</td>
      <td>Y</td>
      <td>Retired</td>
      <td>NaN</td>
      <td>45.0</td>
      <td>540.0</td>
      <td>540.0</td>
      <td>None</td>
      <td>NEVER SMOKED/VAPED</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


```python
hosp_clients_df.shape
```

    (366393, 238)


To separate our data into timelines for each client, we store a dataframe for each client in a dictionary.
- Each key will be a `Clientid`.
- Each value will be a time series indexed dataframe for that 
client.


```python
hosp_clients_dict = {clientid: hosp_clients_df[hosp_clients_df['Clientid'] == clientid].set_index('_Date')
                     for clientid in hosp_clients.tolist()}
```



I have a hunch that hardly any of these columns will change for the time period we have available. The function below 
counts the number of columns that have any change whatsoever for one client.


```python
def total_changes(df):
    """Total all columns that contain more than a single value in given client dataframe."""
    sums = []
    length = len(df)
    for column in df.columns:
        # Skip Admissions column, we need changes that may predict this.
        if column == 'Admissions':
            continue
        total = df[column].value_counts()
        if not total.empty:
            # Check that there is more than one value in the entire column.
            sums.append(total.values[0] != length)
    # Sum of all columns with multiple values.
    return sum(sums)
```

Get the total number of columns containing more than one value 
for all clients by summing the total number of columns with 
changes for each client.


```python
changes = [total_changes(hosp_clients_dict[client]) for client in hosp_clients]
sum(changes)
```


    1284



Find the average number of columns with multiple values.


```python
sum(changes) / len(changes)
```




    0.781973203410475



There is very little variation. Too little. For the average client, not even one variable, out of 237, will change for 
the entire period under consideration. We can't make a prediction if we don't know about any changes. 

As stated above, to predict a rare event like hospitalization, we need very robust data and this doesn't cut it.

Unfortunately, after the long wait for the data and a steeper than anticipated learning curve for using big data tools 
like AWS and Dask, we were nearly six weeks into labs before we fully realized how lacking our data set truly is. 
This is a project worth doing, hopefully others will be able to help make it happen if our partner is able to create a 
richer dataset.

Thanks for listening!