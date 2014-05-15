---
layout: page
title: Physician Affinity Graph
tagline: 
---
{% include JB/setup %}

# Physician Affinity Graph

## Summary
This is an attempt to recreate and expand upon the work done by [Fred
Trotter](www.fredtrotter.com) on a network graph of pysicians and orgainizations
involved in providing health care services to Medicare patients.  The
[DocGraph](www.docgraph.org) was first presented at the
[Strata](http://strata.oreilly.com/2012/11/docgraph-open-social-doctor-
data.html) conference in 2012.  This initative applied network analysis on
health care provider (HCP) relationships that are not formally curated within a
social network like Facebook, LinkedIn or Twitter.  The FOIA enabled Medicare
claims public datasets were used to infer relationships between the HCP
entities. As far as we have researched (Feb 2014), the data nor the
visualization of the graph is publically available yet.

## What do we hope to answer?
In our work, we hope to create the entity relationship graph based on comman
Medicare claim counts (referrals) with additional considerations to shared HCP
specialization, shared drug prescriptions, propensity to prescribe propritory
versus generics and geographic area.  Specifically, our aim is to understand the
strength of the relationship between HCPs that can, in turn, help answer the
followin questions:

1. Who are the most influential doctors for a particular disease area in a
specific geography?
2. Can we apply ranking or an influence score to the healthcare provider?
3. Is there a way, using these data, to develop a 'Key Opinion Leader' KOL
recommendation engine for the health care provider community?

We see payers, pharmaceutical companies, patients, researchers and academia as
potential stakeholders interested in answers to the above questions.

The insights will be aggregated at the following levels:

- Providers (NPI, name, city, state) view (by Geo and Specialty)
- Geography (city and state) view (by HCP and Specialty)
- Specialties view (by Geo and HCP)

This effort is a part of the session end project submission requierments for
[Exploratory Data Analysis](http://malecki.github.io/edav/) and [Machine
Learning for Data Science](http://www1.ccls.columbia.edu/~ansaf/4721/index.html)
courses in the Data Science program at Columbia University, New York.
[Rajesh Madala](www.linkedin.com/pub/rajesh-madala/6/283/336), [Mandeep
Sigh](www.linkedin.com/pub/mandeep-singh-cfa/2/236/214) and [Mayank
Misra](https://www.linkedin.com/in/mayankmisra) worked on this project.

The scope was adjusted in April to focus on the New York, New Jersey,
Connecticut tristate area and the HCPs involved with Oncology as a specialty.

## Data Sources

The following datasets and sources were identified for this analysis:

1. [National Plan and Provider Enumeration System (size 5 GB)](http://nppes
.viva-it.com/NPI_Files.html) data from [Centers for Medicare and Medicaid Servic
es](http://en.wikipedia.org/wiki/Centers_for_Medicare_and_Medicaid_Services)
(CMS).
    - The dataset captures every provider, plan, and clearinghouse involved in
enabling healthcare in the United States.  The [Healthcare Insurance Portability
and Accountability Act (HIPPA) of 1998](http://www.cms.gov/Regulations-and-
Guidance/HIPAA-Administrative-Simplification/HIPAAGenInfo/index.html) requires
every  entity that sends a claim (raises a bill) for providing healthcare to
acquire a NPI.  This number is unique as a social security number and is
referenced across public and private systems to identify a HCP entity.  The
NPPES data is keyed on [National Provider
Identifier](http://en.wikipedia.org/wiki/National_Provider_Identifier) (NPI).
    - It is distributed as a CSV file. It contains one record per covered
entity, which include nurse practitioners, community health centers, hospitals,
and other care providers who are not physicians.  Each record contains a mailing
and the practice address for the HCP.
    - This dataset is available for 2001, 2010 and 2011. [The National Provider
Identifier (NPI): What You Need to Know](http://www.cms.gov/Outreach-and-
Education/Medicare-Learning-Network-MLN/MLNProducts/downloads/NPIBooklet.pdf)
booklet is a good resource to understand the NPI categories.  In the graph
database parlance, this is a node database.
    - NPI information is also available through API calls via
[BloomAPI](http://www.bloomapi.com/documentation) and
[DocNPI](http://docnpi.com/).  BloomAPI is open source, more up-to-date
(4-6months after CMS releases the data) but lacks the ability to look up doctors
by type (e.g. geography or specialization).
    - The file has 314 columns.

2. [Physician Referral Patterns – 2009, 2010, 2011 (size 2
GB)](https://questions.cms.gov/faq.php?faqId=7977) : These files represent 3
years of data showing the number of referrals from one physician to another
(data for 2012 has also been released at the time of publishing).  This dataset
was made available as a FOIA request.  Each record in this database captures the
occurrences where a claim was made for the same patient by two health care
providers (HCP) in a rolling 30 day time period.  The two HCPs are picked up
only if they have had 11 or more 'shared' patients for which Medicare was
billed.  The 11 number is a CMS standard to obfuscate identifying individual
patient by their referral patterns.  A record in this data set is of the form:
{1112223334, 5556667778, 1100}.  The first value in the NPI of the HCP who
provided care the first.  The second value is the NPI who provided care
subsequently.  The third value is the number of time this occurred in the past
rolling 30 days for that year. This dataset, probably due to a technical issues,
was offline for a period of time while this work was carried out. As a fall back
[DocGraph Edge Database v1.0 Open Source](http://notonlydev.com/product
/docgraph-edge-database-v1-0-open-source/) was download for a token of $1.  This
dataset is available for 2011 only.  The documentation accompanying this dataset
further explains the Physician Referral data as follows:
    - If Provider A sees a patient on January 15, and Provider B sees the same
patient on February 15, then that counts as “+1.”
    - If the same patient sees Doctor A on January 15, Doctor B on February 15
and then again on June 15 and July 15, then that counts as “2″ referrals in this
dataset. When a referral relationship has a score of 1,100 we cannot know if
this was 11 patients with 100 referral instances, or 1,100 patients with 1
referral instance, or 10 patients with 10 referral instances and 1 patient with
1,090 referral instances. The whole point here is that we have a score that
approximates the strength of the relationship between two entities in the NPI
database, and for that purpose it does not really matter what kind of patient
flow is being indicated.

3. [Medicare Part D Prescribing Data 2011 & Medicare Part D Prescribing Data
(Patients 65 or Older) 2011 (size 7 GB)](https://projects.propublica.org/data-
store/): These data, made available by [ProPublica](www.propublica.org), include
all drugs prescribed by doctors 11 or more times that year to Part D patients,
including those 65 and older.
    - A lookup file is provided to match unique prescriber ID to a
practitioner's DEA or NPI number or other identifier.  The NPI number will be
used to tie this to the other datasets.
    - This data is available for 2011 only.

4. [Health Care Provider Taxonomy (size 2 MB)](http://codelists.wpc-
edi.com/mambo_taxonomy_2_pre_production.asp). This dataset adds to the profile
of a HCP entity by detailing their specialization (Type, Classification,
Specialization, Definition).
    - It can be linked to NPEES datasets by linking the files through the NPPES
taxonomy code.
    - The CSV can be [downloaded here](http://www.nucc.org/index.php?option=com_
content&view=article&id=107&Itemid=132).

5.  [National Drug Code Directory (25
MB)](http://www.fda.gov/Drugs/InformationOnDrugs/ucm142438.htm): This dataset
was used to add details for a prescribed drug.  The ProPublica prescribing
dataset was mapped to this Drug Directory to obtain information about the
Labeler, Marketing start/end date, form strength etc.
    - The dataset documentation describes a labeler as "A labeler may be either
a manufacturer, including a repackager or relabeler, or, for drugs subject to
private labeling arrangements, the entity under whose own label or trade name
the product will be distributed".
    - Our intent was to use the NDC details to link a physician to a
pharmaceutical company.  We understand that the linking physicians to a labeler
will not provide an accurate picture of this relationship.  We will swap this
dataset in future with one through which we can link NDC to the patent holder or
licensee

## Enabling technologies

- Gephi
- Python
- R:  For analysis we referenced the work done by [Anthony Damico on public data
sets](http://www.asdfree.com/search/label/national%20plan%20and%20provider%20enu
meration%20system%20%28nppes%29) specially in terms of using R code.
- Tableau (maybe)
- Oracle
- Neo4j (maybe)
- GitHub

## Methodology

### Relationships
   - The NPI will be used as the cross reference key to join the datasets.
   - The physician referral dataset will be used to establish the relationship
between HCPs.  Here, the occurrence count field will be used as the weight in
our Machine Learning algorithms.  The more the count, the tighter the
relationship.
   - The NPPES data has the practice and the mailing address for each provider
(HCP).  These will be used to connect the data to a geography.
   - The NPI will also be linked to the ProPublica data set on drug name.  The
lookup file for matching NPI to DEA (prescriber id) will be used to join NPPES
and Propublica datasets.
   - The drug prescriptions will be linked to pharmaceutical/labeler companies
using.  An assumption is made that the prescription history of a HCP is a
indicator of the pharma - physician relationship and that more often than not,
the labeler and the pharmaceutical company are will be the same.
   - For establishing a Speciality, we will use NPPES taxonomy codes and enrich
them using [Health Care Provider Taxonomy](http://codelists.wpc-
edi.com/mambo_taxonomy_2_pre_production.asp). The CSV can be [downloaded here](h
ttp://www.nucc.org/index.php?option=com_content&view=article&id=107&Itemid=132)
 - The [relationship model is here](https://www.lucidchart.com/documents/view/22
114dcc-7568-460b-8719-85262694d7ed)
 - ![Data Source Relationships](https://www.lucidchart.com/publicSegments/view
/5366e1fc-c0b4-4bba-a41e-04cd0a00ca90/image.png "Data Source Relationships")

## Data Wrangling
Most of the datasets are in comma separated values. The files were individually
downloaded and uncompressed.  Cloud storage was used to archive the raw files
and make them available to the team. The semi prepared and final versions of the
data extracts were also uploaded to the cloud.  They have been made public and
are available [here](http://map-the-google-drive-folder-here)

## Initializing requiered libraries

### Unix
Our first attempt at sampling, merging and enriching the dataset were based on
Unix commands.

```
--replace comma in " "( double strings)  with blank
awk -F'"' -v OFS='' '{ for (i=2; i<=NF; i+=2) gsub(",", "", $i) }1'
npidata_20050523-20140309.csv > npidata.csv

-- create a file with npi ,entity_type_code ,provider_last_name
,provider_first_name ,provider_city ,provider_state provider_zipcode
,provider_taxonomy_code columns
cut -d, -f 1,2,6,7,31,32,33,48 < npidata.csv >> npi_data.csv

-- replace comma in " "( double strings)  with blank
awk -F'"' -v OFS='' '{ for (i=2; i<=NF; i+=2) gsub(",", "", $i) }1'
nucc_taxonomy_140.csv > nc_txnmy140.csv

-- create a file with code ,type ,classification ,specialization
cut -d, -f 1,2,3,4,31,32,33,48 < nc_txnmy140.csv >> npi_txnmy140.csv

-- replace comma with ;
sed s/\,/\;/g product.txt >> prod.txt

sed s/\\t/\,/g prod.txt >> prod1.csv

create file with proprietaryname   ,   labelername
cut -d, -f 4,13 prod1.csv >> product.csv
```

The shear size of datafile, the lack of error handling and an inability to
confirm the script workflow did not give us enough confidence in the output.  We
switched to the tried and trusted relational database to store the data and
query the data.  Oracle personal edition was used for this purposes.

### Relational Database (Oracle)
 - Store NPI dataset:
 sqlldr system/Bhaani data = npi_data.csv control = npi_data.ctl
 The Load Control file:

```
OPTIONS (SKIP=1)
load data
truncate
into table npi_tbl
fields terminated by ","
(npi ,
entity_type_code ,
provider_last_name ,
provider_first_name ,
provider_city ,
provider_state ,
provider_zipcode ,
provider_taxonomy_code)
```

 - Store NPI Taxonomy file:
 sqlldr system/Bhaani data = nc_txnmy140.csv control = nc_txnmy140.ctl
 The Load Control file:

```
OPTIONS (SKIP=1)
load data
truncate
into table npi_taxnmy_tbl
fields terminated by ","
TRAILING NULLCOLS
(code ,
type ,
classification ,
specialization )
```

 - Store NDC dataset:
 sqlldr system/Bhaani data = product.csv control = product.ctl
 The Load Control file:

```
OPTIONS (SKIP=1)
load data
truncate
into table product_tbl
fields terminated by ","
TRAILING NULLCOLS
(
proprietaryname   ,
labelername      )
```

 - Store ProPublica Prescriber-toDrug-toClaim dataset
 sqlldr system/Bhaani data = propublica_prescriber_ids_2011.csv control =
propub_prscrbr_id.ctl
 The Load Control file:

```
load data
truncate
into table propublica_ccw_id
fields terminated by ","
TRAILING NULLCOLS
(propub_id ,
bn ,
claim_count ,
claim_count_daw1 ,
claim_count_cmpnd2 ,
quantity_sum ,
day_supply_sum ,
gross_drug_cost_sum )
```

 - Store ProPublica walk over file (that maps ProPublica id to DEA and NPI)
 sqlldr system/Bhaani data = propublica_ccwid_bn_2011_r.csv control =
propub_ccw_id.ctl
 The Load Control file:

```
load data
truncate
into table propub_prscrbr_id
fields terminated by ","
TRAILING NULLCOLS
(propub_id ,
npi  ,
dea1  ,
dea2  ,
dea3 ,
dea4  ,
dea5  ,
dea6  ,
dea7 )
```

 - Store physician_referrals dataset
 sqlldr system/Bhaani data = physician_referrals_2011_100000.csv control =
physician_referral.ctl

## Data Modeling

 - The data extracts for HCPs involved with Oncology in the NY, NJ and CT states
was pulled through:
 NPI with address state in NY, NJ, CT and specialty = oncology:

```
    SELECT  distinct np.npi,
         pr.npi_2,
         pr.ref_cnt
    FROM npi_tbl np,
         npi_taxnmy_tbl nt,
         propub_prscrbr_id ppr,
         propublica_ccw_id pcc,
         (SELECT prop_name, lblr_name
            FROM (SELECT prop_name,
                         lblr_name,
                         lblr_cnt,
                         ROW_NUMBER ()
                         OVER (PARTITION BY prop_name ORDER BY lblr_cnt DESC)
                            rnk
                    FROM (  SELECT LOWER (proprietaryname) prop_name,
                                   INITCAP (labelername) lblr_name,
                                   COUNT (*) lblr_cnt
                              FROM product_tbl
                          GROUP BY LOWER (proprietaryname),
                                   INITCAP (labelername)))
           WHERE rnk = 1) pt,
         physician_referrals pr
   WHERE     np.provider_taxonomy_code = nt.code
         AND np.npi = ppr.npi
         AND ppr.propub_id = pcc.propub_id
         AND LOWER (pcc.bn) = pt.prop_name
         AND np.npi = pr.npi_1
         AND np.provider_state IN ('NJ', 'NY', 'CT')
         AND LOWER (nt.specialization) LIKE '%oncology%'
   ORDER BY npi
```

- This NPI list was used to subset all other datasets.  A unified, flat dataset
was created to facilitate Statistical analysis, Machine Learning algorithms and
visualization.  The SQL used was:

```
  SELECT np.npi,
         np.entity_type_code,
         np.provider_last_name,
         np.provider_first_name,
         np.provider_city,
         np.provider_state,
         np.provider_zipcode,
         np.provider_taxonomy_code,
         nt.classification,
         nt.specialization,
         ppr.propub_id,
         ppr.dea1,
         claim_count,
         claim_count_daw1,
         claim_count_cmpnd2,
         quantity_sum,
         day_supply_sum,
         gross_drug_cost_sum,
         INITCAP (pt.prop_name) proprietary_name,
         pt.lblr_name labeler_name,
         pr.npi_2,
         pr.ref_cnt
    FROM npi_tbl np,
         npi_taxnmy_tbl nt,
         propub_prscrbr_id ppr,
         propublica_ccw_id pcc,
         (SELECT prop_name, lblr_name
            FROM (SELECT prop_name,
                         lblr_name,
                         lblr_cnt,
                         ROW_NUMBER ()
                         OVER (PARTITION BY prop_name ORDER BY lblr_cnt DESC)
                            rnk
                    FROM (  SELECT LOWER (proprietaryname) prop_name,
                                   INITCAP (labelername) lblr_name,
                                   COUNT (*) lblr_cnt
                              FROM product_tbl
                          GROUP BY LOWER (proprietaryname),
                                   INITCAP (labelername)))
           WHERE rnk = 1) pt,
         physician_referrals pr
   WHERE     np.provider_taxonomy_code = nt.code
         AND np.npi = ppr.npi
         AND ppr.propub_id = pcc.propub_id
         AND LOWER (pcc.bn) = pt.prop_name
         AND np.npi = pr.npi_1
         AND np.provider_state IN ('NJ', 'NY', 'CT')
         AND LOWER (nt.specialization) LIKE '%oncology%'
   ORDER BY npi
```

## Exploration and Analysis
========================================================
## Network Analysis of Healthcare Providers based on Medicare Claims data
![Network Analysis of Healthcare Providers based on Medicare Claims data](https:
//dl.dropboxusercontent.com/u/10381353/EdavMLProject/nodeAnalysisScreenshot_2341
36.png "Network Analysis of Healthcare Providers based on Medicare Claims data")

###The data exploration was done using R.  The code is embedded here and the
grpahs images are linked.  An RPubs version is available
[here](http://rpubs.com/mayankmisra/16595)

                ```
{r}
library(ggplot2)
library(GGally)
library(ggmap)
data <- read.csv("npi_pcg_prod.csv")
colnames(data) <- c("State","City","Specialization","Classification","NPI","Type","Name","Labeler","Drug","Ccount","Cqty","Csum")
```
                
###The table below shows the frequency of Claims by State ![Claims by
State](https://dl.dropboxusercontent.com/u/10381353/EdavMLProject/unnamed-
chunk-3.png "Claims by State")

                
```
{r}
table(data[,"State"])

Visualized as a barplot:
{r fig.width=8, fig.height=4}
agg1 <- aggregate(Ccount ~ State , data = data, sum)
p1 <- ggplot(agg1,aes(State,Ccount,fill=State)) 
p1 + geom_bar(stat="identity") + ylab("Claim Count") + ggtitle("Oncology Claims by State")

```
                
###Split by Oncology Specialty Classification ![Oncology Specialty
Classification](https://dl.dropboxusercontent.com/u/10381353/EdavMLProject
/unnamed-chunk-4.png "Oncology Specialty Classification"):


                ```{r fig.width=16, fig.height=6}
agg2 <- aggregate(Ccount ~ Classification, data = data, sum)
p2 <- ggplot(agg2,aes(Classification,Ccount,fill=Classification))  
p2 + geom_bar(stat="identity") + ylab("Claim Count") + ggtitle("Oncology Claims by Classification")
```
                
###Correlation matrix for Claim Count, Quantity and Sum ![CovMatrix](https://dl.
dropboxusercontent.com/u/10381353/EdavMLProject/CovMatrix.png)

                
```{r fig.width=16, fig.height=6}
data_sub <- data[,c(1,10,11,12)]
ggpairs(data=data_sub, 
        columns=2:4, 
        title="Correlation Matrix",
            colour = "State")
    
                
###Distribution of Claim Counts ![Distribution of claim
count](https://dl.dropboxusercontent.com/u/10381353/EdavMLProject/unnamed-
chunk-6.png "Distribution of claim count"):


                      
```{r fig.width=8, fig.height=3}
agg3 <- aggregate(Ccount ~ NPI + State, data = data, sum)
d3 <- density(agg3[,3])
plot(d3,type="n",main="Distribution of Claim Count")
polygon(d3,col="red",border="gray")
                
###Claim Count by Physicians in Tri-State Area  ![Claim Count by Physicians in
Tri-State Area](https://dl.dropboxusercontent.com/u/10381353/EdavMLProject
/unnamed-chunk-7.png "Claim Count by Physicians in Tri-State Area"):


                
```{r fig.width=16, fig.height=6}
agg3 <- agg3[order(agg3$State,agg3$Ccount,decreasing=T),]
p3 <- ggplot(agg3, aes(NPI,Ccount, colour=State)) 
p3 <- p3 + labs(colour = "State") + scale_x_continuous(breaks = c(1,200))
p3 <- p3 + ggtitle("Claim Counts by Physician in TriState Area") + ylab("Claim Count")
p3 + geom_point() + facet_grid(. ~ State) + geom_smooth()
```
                
###Distribution of Claim Quantity ![DistriClaimCount](https://dl.dropboxusercont
ent.com/u/10381353/EdavMLProject/DistriClaimCount.png)

                ```{r fig.width=8, fig.height=3}
agg4 <- aggregate(Cqty ~ Labeler, data = data, sum)
d4 <- density(agg4[,2])
plot(d4,type="n",main="Distribution of Claim Quantity")
polygon(d4,col="red",border="gray")
```
                
### Claim Volume by Top Labelers
![Claim Count by Top
Labelers](https://dl.dropboxusercontent.com/u/10381353/EdavMLProject/unnamed-
chunk-9.png "Claim Count by Top Labelers")


                ```{r fig.width=16, fig.height=6}
agg4 <- agg4[order(agg4$Cqty,decreasing=T),]
agg4_top <- agg4[1:6,]
p4 <- ggplot(agg4_top,aes(Labeler,Cqty,fill=Labeler))  
p4 + geom_bar(stat="identity") + ylab("Claim Quantity") + ggtitle("Claim Quantity by Top Labelers")
```
                
### Claim Amount by Top Labelers
 ![Claim Sum by Top
Labelers](https://dl.dropboxusercontent.com/u/10381353/EdavMLProject/unnamed-
chunk-10.png "Claim Sum by Top Labelers")


                ```{r fig.width=16, fig.height=6}
agg5 <- aggregate(Csum ~ Labeler, data = data, sum)
agg5 <- agg5[order(agg5$Csum,decreasing=T),]
agg5_top <- agg5[1:6,]
p5 <- ggplot(agg5_top,aes(Labeler,Csum,fill=Labeler))  
p5 + geom_bar(stat="identity") + ylab("Claim Sum") + ggtitle("Claim Sum by Top Labelers")
```
                
### Claim Distribution on a map (work in progress):
 ![Claim Distribution on a
map](https://dl.dropboxusercontent.com/u/10381353/EdavMLProject/unnamed-
chunk-11.png "Claim Distribution on a map")


                ```{r fig.width=15, fig.height=6}
freq <- as.data.frame(table(data$City))
longlat <- geocode(unique(as.character(data$City))) 
cities <- cbind(freq, longlat)
colnames(cities) <- c("City","Freq","long","lat")
cities <- cities[order(cities$Freq,decreasing=T),]
top_cities <- cities[1:10,]
states <- map_data("state")
tri_states <- subset(states, region %in% c( "Connecticut", "new jersey", "new york")) 
p <- ggplot(tri_states, aes(x=long, y=lat, group = group)) 
p <- p + geom_polygon(fill="grey10", colour="white") 
p <- p + xlim(-80, -70) 
p <- p + geom_point(data=top_cities, inherit.aes=F, aes(x=long, y=lat, size=Freq), colour="red", alpha=.8) + scale_size(name="Claim Frequency")
p <- p + geom_text(data=top_cities, inherit.aes=F, aes(x=long, y=lat, label=City), vjust=1, colour="red", alpha=.8)
p + ggtitle("Physician Claims in TriState Area")
```
                
## Applying Machine Learning to derive prediction models
========================================================================

### Machine Learnign was done using Python and Scikit learn


    import numpy as np
    import pandas as pd
    import pandas.tools.rplot as rplot
    import pylab as pl
    import pylab as plt
    import scipy as sp
    
    import os,sklearn
    import scipy as sp
    import sklearn.linear_model
    import sklearn.naive_bayes
    from sklearn import metrics
    from sklearn.decomposition import PCA
    from sklearn.cross_validation import train_test_split
    from sklearn.cross_validation import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    from sklearn import datasets, neighbors, linear_model, svm
    from sklearn.cross_validation import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from matplotlib.colors import ListedColormap
    
    %matplotlib inline 
    pd.set_option('display.notebook_repr_html', False)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.max_rows', 25)

## Extract data


    # set current working directory
    path = '/Users/mayank/Downloads/'
    os.chdir(path)



    # Create color maps for 3-class classification problem, as with iris
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])



    # Define features to train on
    features = np.array(['NPI','PROVIDER_CITY', 'PROVIDER_STATE', 'PROVIDER_ZIPCODE', 'PROVIDER_TAXONOMY_CODE', 'CLASSIFICATION','SPECIALIZATION','PROPRIETARY_NAME'])



    # Instantiate classifiers
    knn = neighbors.KNeighborsClassifier(weights='distance')
    logistic = linear_model.LogisticRegression()
    dt = DecisionTreeClassifier(max_depth=None, min_samples_split=1, random_state=0)
    ab = AdaBoostClassifier(dt, n_estimators=300)
    rf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
    sv = svm.SVC(gamma=0.001) 



    #Load Training dataset to a dataframe
    train = pd.read_csv('npi_with_lbl.csv')


## Process Data


    def ProcessData():
        
        train.CLASSIFICATION = pd.factorize(train.CLASSIFICATION)[0]
        train.SPECIALIZATION = pd.factorize(train.SPECIALIZATION)[0]
        train.PROVIDER_CITY = pd.factorize(train.PROVIDER_CITY)[0]
        train.PROVIDER_STATE = pd.factorize(train.PROVIDER_STATE)[0]
        train.PROVIDER_ZIPCODE = pd.factorize(train.PROVIDER_ZIPCODE)[0]
        train.PROVIDER_TAXONOMY_CODE = pd.factorize(train.PROVIDER_TAXONOMY_CODE)[0]
        train.PROPRIETARY_NAME = pd.factorize(train.PROPRIETARY_NAME)[0]
        train.LBL = pd.factorize(train.LBL)[0]
        
    
        X_train, X_test, y_train, y_test = train_test_split(train[features], train.LBL, test_size=0.4)
    
        return X_train, y_train, X_test, y_test

## Measure Performance


    def measure_performance(X,y,clf):
        y_pred=clf.predict(X)   
        print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred)),"\n")
        print ("Classification report")
        print (metrics.classification_report(y,y_pred),"\n")
        print ("Confusion matrix")
        print (metrics.confusion_matrix(y,y_pred),"\n")
        
        if (clf != knn and clf != sv):
            feature_importance = clf.feature_importances_
            feature_importance = 100.0 * (feature_importance / feature_importance.max())
            sorted_idx = np.argsort(feature_importance)
            pos = np.arange(sorted_idx.shape[0]) + .5
            plt.figure()
            plt.barh(pos, feature_importance[sorted_idx], align='center')
            plt.yticks(pos, train[features].columns[sorted_idx])
            plt.title('Variable Importance');
            plt.show()
        
    #     plt.plot(y, y_pred, label='Prediction vs orig')
    #     plt.legend(loc='best')
    #     plt.axis('tight')
    #     plt.title('Plot')
    #     plt.figure()

## Plot Classifiers


    def plot_classifier(X_train,y_train,X_test,y_test,clf):
       
        # Plot the training points
       
        plt.figure()
        ndx = sp.where(y_train == 0)
        plt.plot(X_train[ndx,7], X_train[ndx,3], 'r--', alpha=0.25, label='low count')
        ndx = sp.where(y_train == 1)
        plt.plot(X_train[ndx,7], X_train[ndx,3], 'bs', alpha=0.25, label='medium count')
        ndx = sp.where(y_train == 2)
        plt.plot(X_train[ndx,7], X_train[ndx,3], 'g^', alpha=0.25, label='high count')
        y_pred=clf.predict(X_test)
        ndx = sp.where(y_pred == 0)
        plt.plot(X_test[ndx,7], X_test[ndx,3], 'r--', alpha=1, label='low count')
        ndx = sp.where(y_pred == 1)
        plt.plot(X_test[ndx,7], X_test[ndx,3], 'bs', alpha=1,label='medium count')
        ndx = sp.where(y_pred == 2)
        plt.plot(X_test[ndx,7], X_test[ndx,3], 'g^', alpha=1,label='high count')
    
        plt.xlabel('Labeler Name')
        plt.ylabel('State')
        plt.axis('tight')
        plt.title('2-D plot of training & test data')
        plt.show()
        


    X_train, y_train, X_test, y_test = ProcessData()


### PCA


    pca = PCA(n_components=2)
    pca.fit(X_train)
    X_reduced = pca.transform(X_train)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_train)





    <matplotlib.collections.PathCollection at 0x10bb07810>




![png](PhysicianAffinityGraph_files/PhysicianAffinityGraph_45_1.png)


![PCA](https://dl.dropboxusercontent.com/u/10381353/EdavMLProject/ML-
PAG_45_1.png)


    #k_means = KMeans(n_clusters=3, random_state=0) 
    #k_means.fit(X_train)
    #y_pred = k_means.predict(X_test)
    #plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred);


### Decision Tree


    print('Decision Tree: %f ' %cross_val_score(dt, X_train, y_train).mean())
    dt.fit(X_train,y_train)
    measure_performance(X_test,y_test,dt)


    Decision Tree: 0.737710 
    ('Accuracy:0.747', '\n')
    Classification report
    ('             precision    recall  f1-score   support\n\n          0       0.88      0.84      0.86      1006\n          1       0.07      0.07      0.07        72\n          2       0.14      0.20      0.16        89\n\navg / total       0.77      0.75      0.76      1167\n', '\n')
    Confusion matrix
    (array([[849,  54, 103],
           [ 57,   5,  10],
           [ 63,   8,  18]]), '\n')



![png](PhysicianAffinityGraph_files/PhysicianAffinityGraph_49_1.png)


![DT](https://dl.dropboxusercontent.com/u/10381353/EdavMLProject/ML-
PAG_48_1.png)

### Random Forest


    print('Random Forest: %f ' %cross_val_score(rf, X_train, y_train).mean())
    rf.fit(X_train,y_train)
    measure_performance(X_test,y_test,rf)


    Random Forest: 0.782284 
    ('Accuracy:0.771', '\n')
    Classification report
    ('             precision    recall  f1-score   support\n\n          0       0.86      0.89      0.87      1006\n          1       0.00      0.00      0.00        72\n          2       0.07      0.07      0.07        89\n\navg / total       0.75      0.77      0.76      1167\n', '\n')
    Confusion matrix
    (array([[894,  40,  72],
           [ 66,   0,   6],
           [ 78,   5,   6]]), '\n')



![png](PhysicianAffinityGraph_files/PhysicianAffinityGraph_52_1.png)


![RF](https://dl.dropboxusercontent.com/u/10381353/EdavMLProject/ML-
PAG_50_2.png)

### Ada Boost


    print('Ada Boost: %f ' %cross_val_score(ab, X_train, y_train).mean())
    ab.fit(X_train,y_train)
    measure_performance(X_test,y_test,ab)


    Ada Boost: 0.740571 
    ('Accuracy:0.746', '\n')
    Classification report
    ('             precision    recall  f1-score   support\n\n          0       0.88      0.84      0.86      1006\n          1       0.07      0.07      0.07        72\n          2       0.14      0.20      0.16        89\n\navg / total       0.77      0.75      0.76      1167\n', '\n')
    Confusion matrix
    (array([[848,  55, 103],
           [ 57,   5,  10],
           [ 63,   8,  18]]), '\n')



![png](PhysicianAffinityGraph_files/PhysicianAffinityGraph_55_1.png)


![ADB](https://dl.dropboxusercontent.com/u/10381353/EdavMLProject/ML-
PAG_52_1.png)

### K Nearest Neighbors


    print('KNN: %f ' %cross_val_score(knn, X_train, y_train).mean())
    knn.fit(X_train,y_train)
    measure_performance(X_test,y_test,knn)
    plot_classifier(X_train,y_train,X_test,y_test,knn)


    KNN: 0.784568 
    ('Accuracy:0.780', '\n')
    Classification report
    ('             precision    recall  f1-score   support\n\n          0       0.86      0.89      0.88      1006\n          1       0.02      0.01      0.02        72\n          2       0.13      0.11      0.12        89\n\navg / total       0.75      0.78      0.77      1167\n', '\n')
    Confusion matrix
    (array([[899,  43,  64],
           [ 68,   1,   3],
           [ 75,   4,  10]]), '\n')



![png](PhysicianAffinityGraph_files/PhysicianAffinityGraph_58_1.png)


![KNN](https://dl.dropboxusercontent.com/u/10381353/EdavMLProject/ML-
PAG_54_1.png)

### Support Vector Machine


    print('SVM: %f ' %cross_val_score(sv, X_train, y_train).mean())
    sv.fit(X_train,y_train)
    measure_performance(X_test,y_test,sv)


    SVM: 0.840579 
    ('Accuracy:0.850', '\n')
    Classification report
    ('             precision    recall  f1-score   support\n\n          0       0.86      0.99      0.92      1006\n          1       0.00      0.00      0.00        72\n          2       0.09      0.01      0.02        89\n\navg / total       0.75      0.85      0.79      1167\n', '\n')
    Confusion matrix
    (array([[991,   6,   9],
           [ 71,   0,   1],
           [ 86,   2,   1]]), '\n')



    


    
