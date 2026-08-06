# 

**[Categories of Machines Available	2](#categories-of-machines-available)**

[The Jubail HPC (Abu Dhabi)	2](#the-jubail-hpc-\(abu-dhabi\))

[C2 Qos Nodes (CPU and GPU)	2](#c2-qos-nodes-\(cpu-and-gpu\))

[C2 Reservation (CPU only)	2](#c2-reservation-\(cpu-only\))

[Kindi Machine (Abu Dhabi)	2](#kindi-machine-\(abu-dhabi\))

[The Green HPC (New York City)	2](#the-green-hpc-\(new-york-city\))

[**Which Machine To Use?	2**](#which-machine-to-use?)

[Machines That Can be Used by Any Member	2](#machines-that-can-be-used-by-any-member)

[Machines That Can be Used by Certain Members Only	3](#machines-that-can-be-used-by-certain-members-only)

[**Jubail HPC Nodes (C2 Qos, C2 Reservation)	4**](#jubail-hpc-nodes-\(c2-qos,-c2-reservation\))

[General Guidelines About Using the HPC Nodes	4](#general-guidelines-about-using-the-hpc-nodes)

[Training	4](#training)

[Software Installation	4](#software-installation)

[Data Storage	4](#data-storage)

[General Guidelines	5](#general-guidelines)

[How To Access Different Node Types	5](#how-to-access-different-node-types)

[Access the Login Nodes of the HPC	5](#access-the-login-nodes-of-the-hpc)

[C2 Qos Nodes (GPU Nodes)	5](#c2-qos-nodes-\(gpu-nodes\))

[**C2 Reservation Nodes (CPU Nodes)	6**](#c2-reservation-nodes-\(cpu-nodes\))

[**Kindi Machine	6**](#kindi-machine)

[Difference between El Kindi and the HPC	6](#difference-between-el-kindi-and-the-hpc)

[How to Access the Kindi Machine	6](#how-to-access-the-kindi-machine)

[General Guidelines About Using the Kindi Machine	7](#general-guidelines-about-using-the-kindi-machine)

[Data Management	7](#data-management)

[Installing Software	7](#installing-software)

[Information About the Machine	8](#information-about-the-machine)

[**Getting Help	8**](#getting-help)

# 

# 

# **Categories of Machines Available** {#categories-of-machines-available}

There are 3 machines available to the members of our team:

## The Jubail HPC (Abu Dhabi) {#the-jubail-hpc-(abu-dhabi)}

This is an HPC in Abu Dhabi (NYU Abu Dhabi). By default, you should use this HPC. There are two categories of nodes that our members have the right to access on this HPC. They are described below:

### C2 Qos Nodes (CPU and GPU) {#c2-qos-nodes-(cpu-and-gpu)}

This is a subset of the HPC nodes. Our team has a priority accessing these nodes. These nodes have both CPUs and GPUs.

### C2 Reservation (CPU only) {#c2-reservation-(cpu-only)}

This is a subset of the HPC nodes reserved exclusively to our team (16 CPU nodes).

Note that we are discouraged from using the other HPC nodes (other than the above two categories of nodes). Usually, our team is encouraged to use only the two above categories. If you need to use other nodes than those in the above two categories, let me know and I can try to get permission from the HPC admins.

## Kindi Machine (Abu Dhabi) {#kindi-machine-(abu-dhabi)}

This is a separate machine that has 8xA100 GPUs. It is not a part of the NYUAD HPC. It is owned by our team.

## The Green HPC (New York City) {#the-green-hpc-(new-york-city)}

This is an HPC in New York City. We can use some nodes of that HPC, but usually, the nodes are a bit busy. Feel free to use nodes from that HPC if you find some free nodes. But because there are not a lot of free nodes, we usually do not use this HPC a lot (unless our HPC is busy).

Full documentation of the Green HPC and how to use it is here: [Link](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene). 

# **Which Machine To Use?** {#which-machine-to-use?}

## Machines That Can be Used by Any Member {#machines-that-can-be-used-by-any-member}

* Whether you are an NYUAD student or not, by default any member has the right to use the “C2 Qos Nodes”. You can use these nodes to do work on CPU or on GPU. By default, we give you access to the C2 Qos, but we might forget to do that sometimes though. So if you try to access these nodes and get an access denied error, let us know and we will add you.

## Machines That Can be Used by Certain Members Only {#machines-that-can-be-used-by-certain-members-only}

* Users working on certain projects can access the Kindi machine or the C2 Reservation.

# 

# **Jubail HPC Nodes (C2 Qos, C2 Reservation)** {#jubail-hpc-nodes-(c2-qos,-c2-reservation)}

## General Guidelines About Using the HPC Nodes {#general-guidelines-about-using-the-hpc-nodes}

### Training {#training}

- Please read the following short HPC guide (takes only 30mn): [https://crc-docs.abudhabi.nyu.edu/hpc/training/index.html\#training](https://crc-docs.abudhabi.nyu.edu/hpc/training/index.html#training)   
- Please check the main page of the HPC to explore what are the resources available to you: [https://crc-docs.abudhabi.nyu.edu/hpc/hpc.html](https://crc-docs.abudhabi.nyu.edu/hpc/hpc.html) 

### Software Installation {#software-installation}

- Installing software system-wide (i.e., for all the users) using root access is usually hard on an HPC node. This is mainly because it creates problems for other users.  
- It is recommended to install all software locally (i.e., for the user only). There are two recommended ways for doing so:  
  - By using anaconda (or miniconda which is a light version of anaconda).  
  - By compiling the software from source to install it locally.  
- We recommend using miniconda. It is already preinstalled in the HPC. Please read the following guide on how to setup it up and use it (this will install it locally in your account): [https://crc-docs.abudhabi.nyu.edu/hpc/software/hpc\_miniconda.html](https://crc-docs.abudhabi.nyu.edu/hpc/software/hpc_miniconda.html)   
- The following tutorials explain how to use anaconda (same as miniconda)  
  - Video presenting Anaconda: [https://www.youtube.com/watch?v=YJC6ldI3hWk](https://www.youtube.com/watch?v=YJC6ldI3hWk)   
  - Anaconda Tutorial: [https://linuxhint.com/anaconda-python-tutorial/](https://linuxhint.com/anaconda-python-tutorial/)   
- Pytorch is already installed on the HPC, you can just load it. Here is a guide on how to do so: [https://crc-docs.abudhabi.nyu.edu/hpc/software/hpc\_pytorch.html](https://crc-docs.abudhabi.nyu.edu/hpc/software/hpc_pytorch.html)   
  The previous guide shows how one can install pytorch-1.4, but a different version of pytorch might be available when you start working. You can find the list of miniconda environments available by typing   
  	conda env list  
  Look for the available version of pytorch. At the time of writing this guide, pytorch-1.11.0 was available.  
- A guide about installing Tiramisu on the HPC: [https://docs.google.com/document/d/1de9q-x82zsiI0uxKzrB3g89eTt58QmZBsuIzzmxAD4s/edit?usp=sharing](https://docs.google.com/document/d/1de9q-x82zsiI0uxKzrB3g89eTt58QmZBsuIzzmxAD4s/edit?usp=sharing) 

### Data Storage {#data-storage}

- You are advised to use the folder **/scratch/\<NetID\>** to host your data. For example, if your NETID is rb4792, you can use the folder /scratch/rb4792. If this folder does not exist, you can create it. You can use this folder to host all of your files. Note that files stored in the home directory, or in other directories, other than /scratch, can’t be used from C2 nodes (because C2 nodes are compute nodes). Note also that data unused for more than 90 days, is deleted, so keep using your data, otherwise it might be deleted.  
- A full guide on data storage is provided here (this is the same general HPC guidance on how to store data): [https://crc-docs.abudhabi.nyu.edu/hpc/storage/index.html](https://crc-docs.abudhabi.nyu.edu/hpc/storage/index.html) 

### General Guidelines {#general-guidelines}

- Use the “screen” tool to save your session when you close your shell (or if there is an internet issue). With screen, your session does not close if the internet is lost or if your shell is closed. This is useful for long term training. Here is an example of a [tutorial](https://linuxize.com/post/how-to-use-linux-screen/) about screen (it should be already installed).

## How To Access Different Node Types  {#how-to-access-different-node-types}

### Access the Login Nodes of the HPC {#access-the-login-nodes-of-the-hpc}

- If you do not have access to the HPC, please follow the steps [here](https://docs.google.com/document/d/1IlgR6cSv7k76WZz1zHDKkPofWd9A9GHTZizubfxbk9k/edit?usp=sharing) to get access.  
- To access the HPC (to access the login nodes of the HPC):  
  - Method 1: By connecting to the NYUAD VPN (the NYUNY VPN will not work)  
    - Connect the NYUAD VPN  
    - Connect to Jubail:   
      - ssh \<NetID\>@jubail.abudhabi.nyu.edu  
  - Method 2: Without connecting to VPN, and instead connecting to the bastion host  
    - Connect to Bastion host  
      - ssh \<NetID\>@hpc.abudhabi.nyu.edu \-p 4410  
    - Connect to Jubail  
      - ssh \<NetID\>@jubail.abudhabi.nyu.edu  
- After following one of the above methods, you’ll be directed to the login nodes of the HPC. These are not the nodes that you’ll be using to do your computations. These are just the gates of the HPC. From these nodes you should login to the compute nodes as described below.  
- More details about how to access here: [https://crc-docs.abudhabi.nyu.edu/hpc/system/access\_jubail.html](https://crc-docs.abudhabi.nyu.edu/hpc/system/access_jubail.html) 

### C2 Qos Nodes (GPU Nodes) {#c2-qos-nodes-(gpu-nodes)}

- These are two GPU nodes that our team has priority to use.  
- These are two GPU nodes (cn009,cn023), one with two A100 GPU cards (cn023) and the other one with one A100 GPU card (cn009).  
- These GPU nodes are shared with other research teams, so they might be full sometimes. Once you submit your request you might need to wait a bit. If the waiting time is too long, please let me know.  
- You need to have access to the “C2 Qos” to use these nodes, if you do not have access to the C2 Qos, ask Riyadh to grant you access.  
- To use them  
  - First, you need to access to the login nodes of the HPC, as described in the section (Access the Login Nodes of the HPC).  
  - Second, you need to access the C2 Qos  nodes, use slurm as follows (access takes some time)  
    **srun \--pty \-n1 \-q c2 \-p nvidia \--gres=gpu:a100:1 bash**

- Our A100 GPUs have 80GB of RAM. If you need 80GB of RAM but the nodes that you get using the above command does returns a nodes with a smaller RAM, let us know.  
- The above command will give you access to one of the nodes. If you need to login to a particular nodes (cn009 or cn023), you need to add the following option to the above command:

		**\-w cn009** 

### C2 Reservation Nodes (CPU Nodes) {#c2-reservation-nodes-(cpu-nodes)}

- These nodes are CPU nodes that are exclusive to our team. They are compute nodes.  
- These nodes are only accessible to certain projects since they have experiments that require an isolated environment.  
- You need to have access to the C2 Reservation to use these nodes. If you do not have access to the C2 Reservation, ask Riyadh to grant you access.  
- To use them:  
  - First, you need to access to the login nodes of the HPC, as described in the section (Access the Login Nodes of the HPC).  
  - Second, you need to access the C2 reservation nodes, use slurm as follows (access takes some time)  
    **srun \--pty \-n1 \--reservation=c2 \--nice=2000 bash** 

### Limits and Waiting Time for Job Requests on Jubail

If you submit jobs to Jubail and the waiting time is high, check the following to figure out why (and to try to solve the problem).

The **NODELIST(REASON)** column from the [**squeue**](https://crc-docs.abudhabi.nyu.edu/hpc/jobs/quick_start.html#basic-slurm-commands) command can be used to give a better idea of why a job is pending, some of the most common **pending** reasons are as follows:

* **Resources:** means the requested resources are not available.  
* **Priority:** means there are other jobs pending in the queue with higher priority than this job, there are many factors for defining priority like when the job is submitted, the requested resources and few others.  
* **QOSGrpGRES**: means the **c2** condo as it only has access to **5 GPUs**, those are currently used and it reaches its limit.  
* **QOSMaxWallDurationPerJobLimit:** The requested time limit is exceeding the the maximum time limit for this partition as mentioned [here](https://crc-docs.abudhabi.nyu.edu/hpc/jobs/quick_start.html#partitions-summary).  
  *  The job will be queued for this reason and will not be able to be executed at all, so please try to **eliminate** the time limit and resubmit the job.  
  * Otherwise, If the job can not be completed in the maximum allowed time, please submit the job with maximum allowed time then contact us with a notice period of **24 hours** within our working days for time extension.

This [link](https://slurm.schedmd.com/job_reason_codes.html) can be helpful in understanding the different pending reasons.

**General notes:**

1. **show-my-limits** can be used from any of the login nodes to check the latest **limits** of the HPC account for the different partitions.  
2. **gpuload** can be used to check the **GPU** load on the machine.  
3. **show-my-condo** can be used to check the current usage of your condo/s.  
4. This [link](https://crc-docs.abudhabi.nyu.edu/hpc/hpc_load/index.html) from our wiki can be useful in getting the real time **HPC Load** of the cluster.

# **Kindi Machine** {#kindi-machine}

## Difference between El Kindi and the HPC {#difference-between-el-kindi-and-the-hpc}

El Kindi machine is not a part of the NYUAD HPC, it is managed separately, but most of the rules of using the cluster apply also to el kindi. There are a few differences:

- We do not use slurm to submit jobs on el Kindi while you need to use slurm to submit jobs on the HPC.  
- El Kindi does not have access to the HPC file system /scratch, /works, … It has it own disks.  
- El Kindi does not have access to the software available for the HPC. Therefore you can’t load the preinstalled HPC miniconda and load its environments. You have to install your own miniconda and create your own environment. 

## How to Access the Kindi Machine {#how-to-access-the-kindi-machine}

You can access the machine using SSH.  
Server : [**kindi.abudhabi.nyu.edu**](http://kindi.abudhabi.nyu.edu)  
Port : **4410**

To use SSH to access the machine, first make sure to connect to the NYUAD VPN, then run the following command

ssh   \-p 4410   ***\<NYUAD-NETID\>***@kindi.abudhabi.nyu.edu

Make sure you replace ***\<NYUAD-NETID\>*** with your NETID (e.g., rb4792).

More about accessing the server through SSH, Please follow this KB  [https://goo.gl/eg99HR](https://goo.gl/eg99HR)

## General Guidelines About Using the Kindi Machine {#general-guidelines-about-using-the-kindi-machine}

- Please avoid using the CPUs of the machine for DNN training. You can still use the CPUs for data preprocessing though.  
- Use the “screen” tool to save your session when you close your shell (or if there is an internet issue). With screen, your session does not close if the internet is lost or if your shell is closed. This is useful for long term training. Here is an example of a [tutorial](https://linuxize.com/post/how-to-use-linux-screen/) about screen (it should be already installed).

## Data Management {#data-management}

- The machine has 6 disks. 2 are 3.84 TB SSD's and the remaining 4 are NVME.  
  - SSD disks are configured as RAID 1 and mounted to **/** and **/home** partition.  
  - The 4 NVME disks have 14TB of free space. They are mounted to /data.  
- All of the disks have an automatic backup system to backup your data.  
- You are advised to create a folder in /data with your NETID as the name of the folder. For example, if your NETID is rb4792, you can create the folder /data/rb4792. You can use this folder to host all of your files.  
- Please avoid using your home directory to store large files. Any large file should be stored in your /data folder. Your home directory should only be used for storing small files.  
- Please make sure to delete any files that you do not need since the 14TB space is not enough for a large number of users.

## Installing Software {#installing-software}

- Installing software system-wide (i.e., for all the users) using root privileges is very hard on an HPC node. This is mainly because it creates problems for other users. So if you try to install software and the software is asking for sudo or root privileges, then you are going in the wrong direction.  
- It is instead recommended to install all software locally (i.e., for the user only). There are two recommended ways for doing so:  
  - By using Anaconda.  
    - Video presenting Anaconda: [https://www.youtube.com/watch?v=YJC6ldI3hWk](https://www.youtube.com/watch?v=YJC6ldI3hWk)   
    - Anaconda Tutorial: [https://linuxhint.com/anaconda-python-tutorial/](https://linuxhint.com/anaconda-python-tutorial/)   
    - To install Anaconda and Pytorch for example, you can follow the instructions on [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)   
  - By compiling the software from source to install it locally.  
- A guide about using Jupiter notebook on El Kindi: ​​[https://docs.google.com/document/d/1ND1mWCUJPaylLsjLy-Q-eW2P2mWsQJLjYSwnizVwwxY/edit?usp=sharing](https://docs.google.com/document/d/1ND1mWCUJPaylLsjLy-Q-eW2P2mWsQJLjYSwnizVwwxY/edit?usp=sharing)   
- A guide about installing Tiramisu on the HPC (this guide is not designed for el Kindi but might help): [https://docs.google.com/document/d/1de9q-x82zsiI0uxKzrB3g89eTt58QmZBsuIzzmxAD4s/edit?usp=sharing](https://docs.google.com/document/d/1de9q-x82zsiI0uxKzrB3g89eTt58QmZBsuIzzmxAD4s/edit?usp=sharing) 

## Information About the Machine {#information-about-the-machine}

- CPU  
  - CPU: AMD EPYC 7742 2.25GHz 64-Core processor.  
  - RAM: 16x64GB Dual Rank x4 DDR4 RAM  
- GPUs  
  - The machine has 8 Nvidia A100 GPUs (each has 80GB of global memory).  
- Nvidia drivers and cuda (version 11\. 4\) are already installed.  
- To find the driver version and inspect the status of the GPUs, run the command  
  	nvidia-smi

# **Getting Info About the HPC**

- You can check your limits on your account using the command: show-my-limits   
- To see the status of the C2 condo (i.e., how many C2 GPUs are used; not all users may have access to this command ): show-my-condo

# **Getting Help** {#getting-help}

- If you have an issue with your NYUAD account or VPN, please send an email to nyuad.it@nyu.edu     
- If you have an issue related to the HPC, please contact j[ubail.admins@nyu.edu](mailto:jubail.admins@nyu.edu)  
- If you have an issue with El Kindi machine please contact [nyuad.it@nyu.edu](mailto:nyuad.it@nyu.edu) 

