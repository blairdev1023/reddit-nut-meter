## Setup EMR and Jupyter Notebook w/ Spark

#### Create EMR cluster

```
bash launch_cluster.sh s3_bucket mypem 6 cluster_name
```

Notes:
launch_cluster.sh is a file in ~/scripts
mypem should be your pem file w/out .pem
s3_bucket is the name of your bucket
6 is the number of ec2 'workers
*** WTF IS CLUSTER_NAME ***


#### Copy jupyspark-emr.sh to your EMR

```
scp ~/scripts/jupyspark-emr.sh hadoop@host_name:~/.
```

host_name is the HostName in the ~/.ssh/config file

#### SSH into EMR

In the .ssh file, open config and add your EMR

Still in .ssh, run:
```
ssh shortcut_name_of_emr
```

Note: this is the name you specified in your config file

#### Run jupyspark-emr.sh in EMR

```
pip install awscli

aws configure --> add your aws credentials including region (I use us-east-1)

bash jupyspark-emr.sh
```

#### Create Secure SSH Tunnel

```
ssh -NfL 8886:localhost:8889 shorcut_name_of_emr
```

Note: 8887 is the port specified in jupyspark-emr.sh
      8886 is just a local port open on my machine
      shorcut_name_of_emr is the Host alias used in the config file

 in browser enter: localhost:8886

 Note: 8887 is the portal specified in your jupyspark-emr.sh script

### Go to http://0.0.0.0:8886 and BOOM!
