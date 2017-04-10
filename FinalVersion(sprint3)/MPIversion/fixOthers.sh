#!/bin/bash
#ALL_HOSTS=(gv2017zvg000000 gv2017zvg000001 gv2017zvg00000A gv2017zvg00000C gv2017zvg000003 gv2017zvg000004 gv2017zvg000005 gv2017zvg000007 gv2017zvg000008 gv2017zvg000009)
#ALL_HOSTS=(gv2017zvg000001 gv2017zvg00000A gv2017zvg00000C gv2017zvg000003 gv2017zvg000004 gv2017zvg000005 gv2017zvg000007 gv2017zvg000008 gv2017zvg000009)
ALL_HOSTS=(azure1 azure2 azure3 azure4 azure5 azure6 azure7 azure8 azure9 azure10)
#ALL_HOSTS=(gv2017zvg000000 gv2017zvg00000A gv2017zvg00000C gv2017zvg000003 gv2017zvg000004 gv2017zvg000005 gv2017zvg000007 gv2017zvg000008 gv2017zvg000009)

#do this for the master vm
#sudo apt-get install nfs-kernel-server
#sudo echo "export /home/gvuser/MPIversion" >> /etc/exports
#sudo systemctl start nfs-kernel-server.service

#option y outputs the public key
#ssh-keygen -y -f ~/.ssh/id_rsa > ~/.ssh/id_rsa.pub

#By default, the SSH server denies password-based login for root. In /etc/ssh/sshd_config, change:
#PermitRootLogin without-password
#to
#PermitRootLogin yes
#And restart SSH:

#sudo service ssh restart

#ssh $HOST -t 'ssh-keygen -t rsa;cat ~/.ssh/id_rsa.pub | ssh gv2017zvg000000 "cat >> .ssh/authorized_keys";cat ~/.ssh/id_rsa.pub | ssh gv2017zvg000001 "cat >> .ssh/authorized_keys";cat ~/.ssh/id_rsa.pub | ssh gv2017zvg000003 "cat >> .ssh/authorized_keys";cat ~/.ssh/id_rsa.pub | ssh gv2017zvg000004 "cat >> .ssh/authorized_keys";cat ~/.ssh/id_rsa.pub | ssh gv2017zvg000005 "cat >> .ssh/authorized_keys";cat ~/.ssh/id_rsa.pub | ssh gv2017zvg000007 "cat >> .ssh/authorized_keys";cat ~/.ssh/id_rsa.pub | ssh gv2017zvg000008 "cat >> .ssh/authorized_keys";cat ~/.ssh/id_rsa.pub | ssh gv2017zvg000009 "cat >> .ssh/authorized_keys";cat ~/.ssh/id_rsa.pub | ssh gv2017zvg00000A "cat >> .ssh/authorized_keys";cat ~/.ssh/id_rsa.pub | ssh gv2017zvg00000C "cat >> .ssh/authorized_keys";exit'

TARGETS=("${ALL_HOSTS[@]}")
for HOST in "${TARGETS[@]}"; do
	#ssh $HOST -t 'mkdir stage2ZerosRemoved;sudo mount gv2017zvg000009:/home/gvuser/stage2ZerosRemoved /home/gvuser/stage2ZerosRemoved;exit'
	#ssh $HOST -t 'mkdir stage2;mkdir stage2half1;mkdir stage2half2;exit'
	#ssh $HOST -t 'sudo apt-get -y install python3-pip;pip3 install --user numpy;pip3 install --upgrade pip;pip3 install --user pandas;pip3 install --user pydicom;pip3 install --user scikit-image;sudo apt-get -y install python3-tk;pip3 install --user numpy.stl;pip3 install --user sklearn;exit'
	#ssh $HOST -t 'sudo apt-get -y install python3-pip;sudo pip3 install numpy;sudo pip3 install --upgrade pip;sudo pip3 install pandas;sudo pip3 install pydicom;sudo pip3 install scikit-image;sudo apt-get -y install python3-tk;sudo pip3 install numpy.stl;sudo pip3 install sklearn;exit'
	#ssh $HOST -t 'printenv LD_LIBRARY_PATH;exit'
	#sudo apt-get install -y python3-pip
	#sudo apt-get install -y python3-tk

	#ssh $HOST -t 'pip3 install numpy;pip3 install --user pandas;pip3 install --user scikit-image;pip3 install --user pydicom'
	#ssh $HOST -t 'sudo echo "LD_LIBRARY_PATH="/usr/lib64/openmpi/lib:/home/gvuser/MPIversion/artifacts:/home/gvuser/MPIversion/artifacts:/home/gvuser/MPIversion:/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64_lin" >> /etc/environment;exit'
	#ssh $HOST -t 'printenv PATH;exit'
	#ssh $HOST -t 'locate libmkl_intel_ilp64.so;exit'
	#ssh $HOST -t 'mkdir MPIversion;exit'
	#ssh $HOST -t 'mkdir KagglePre2;exit'
	#ssh $HOST -t 'ssh -t gv2017zvg000000 "exit"'
	ssh $HOST -t 'cd MPIversion;./makeLinks.sh;exit'
	#ssh $HOST -t 'cd stage2;rm *;exit'
	#ssh $HOST -t 'mkdir stage1half2;sudo mount gv2017zvg000004:/home/gvuser/stage1half2 /home/gvuser/stage1half2;exit'
	#ssh $HOST -t 'sudo mount gv2017zvg000007:/home/gvuser/stage2half2 /home/gvuser/stage2half2;exit'
	#ssh $HOST -t 'sudo mount gv2017zvg000000:/home/gvuser/MPIversion /home/gvuser/MPIversion;exit'
	#ssh $HOST -t 'sudo mount gv2017zvg000001:/home/gvuser/KagglePre2 /home/gvuser/KagglePre2;exit'
	#ssh $HOST -t 'sudo umount -f /home/gvuser/MPIversion;exit'
	#ssh $HOST -t 'sudo apt-get -y install nfs-kernel-server;sudo systemctl start nfs-kernel-server.service'
	#ssh $HOST -t 'cd .ssh;chmod 600 id_rsa.pub;exit'
	#ssh $HOST -t 'ssh -t gv2017zvg000000 "exit";ssh -t gv2017zvg000001 "exit";ssh -t gv2017zvg00000A "exit";ssh -t gv2017zvg00000C "exit";ssh -t gv2017zvg000003 "exit";ssh -t gv2017zvg000004 "exit";ssh -t gv2017zvg000005 "exit";ssh -t gv2017zvg000007 "exit";ssh -t gv2017zvg000008 "exit";ssh -t gv2017zvg000009 -t "exit";exit'
	#ssh $HOST -t 'sudo apt-get -y update --fix-missing;exit'
	#ssh $HOST -t 'sudo apt-get -y install libopenmpi-dev;exit'
	#ssh $HOST -t 'sudo apt-get -y install libopenblas-dev;exit'
	#ssh $HOST -t 'sudo apt-get -y install nfs-common;exit'
	#ssh $HOST -t 'echo "export LD_LIBRARY_PATH=/home/gvuser/MPIversion" >> ~/.bashrc;exit'
	#ssh $HOST -t 'echo "export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:/home/gvuser/MPIversion/artifacts:/home/gvuser/MPIversion/artifacts:/home/gvuser/MPIversion:/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64_lin" >> ~/.bash_profile;exit'
	#ssh $HOST -t 'sudo echo "export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:/home/gvuser/MPIversion/artifacts:/home/gvuser/MPIversion/artifacts:/home/gvuser/MPIversion:/opt/intel/mkl/lib/intel64_lin" >> /root/.bashrc;exit'
	#ssh $HOST -t 'echo "export PATH=/home/gvuser/bin:/home/gvuser/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/opt/intel/mkl/lib/intel64_lin" >> ~/.bashrc;exit'
	#ssh $HOST -t 'sudo apt-get -y update --fix-missing;exit'
	#ssh $HOST -t 'sudo apt-get -y upgrade;exit'
	#ssh $HOST -t 'sudo reboot now;exit'
done
#gvsu2017
