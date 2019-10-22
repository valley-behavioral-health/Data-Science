# Exploration of Valley Behavioral Health dataset.

Unfortunately our partners were unable to successful anonymize clinician notes and did not engage with our offers to help 
them do so.

We planned to use each patient's data as a timeline to be consumed by an LSTM or similar model, 
perhaps incorporating reinforcement learning as well. Automatic daily updates with new clinical data would feed into our 
deployed model to give updated predictions. Ideally, we would be able to incorporate our model into the partner's 
existing electronic health record (EHR). Clinicians would otherwise need to input data into our system as well 
as the EHR.

We need to know how many clients we have and how many times they've been hospitalized.

```python
uni = df['Clientid'].unique()
len(uni)
```

    70083

A good number of unique clients.

Find the total number of hospital admissions for the whole dataset.

```python
admins = df['Admissions'].value_counts()
admins.compute()
```

    0.0    15023382
    1.0        3538
    Name: Admissions, dtype: int64

This isn't exactly needle-in-a-haystack territory. It's probably doable if we have robust data.

### Next, we'll take a look at some individual client timelines and try to get a better understanding of the changes that occur.

First, we'll filter for hospitalized patients.


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

Then, get list of hospitalized `Clientid`s and filter the whole dataset to get a dataframe of only those patients.

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

Following my hunch that hardly any of these columns will change for the time period we have available, I made the
function below to count the number of columns that have any change whatsoever for one client.

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

Now we can get the total number of columns containing more than one value per client.

```python
changes = [total_changes(hosp_clients_dict[client]) for client in hosp_clients]

```python
sum(changes) / len(changes)
```

    0.781973203410475

As you can see, there is very little variation in our data. Too little. For the average hospitalized client, not even 
one variable, out of 237, will change for the entire period under consideration. We can't make a prediction if we don't 
know about any changes. 

As stated above, to predict a rare event like hospitalization, we need very robust data and this doesn't cut it.

Unfortunately, after the long wait for the data and a steeper than anticipated learning curve for using big data tools 
like AWS and Dask, we were nearly six weeks into labs before we fully realized how lacking our data set truly is. 
Hopefully  our partner is able to create a richer dataset because this is certainly a project worth doing.

Thanks for listening!