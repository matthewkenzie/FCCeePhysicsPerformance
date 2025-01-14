Author: Suchita Kulkarni
Contact: suchita.kulkarni@cern.ch
This folder will allow you to create your own madgraph sample for heavy neutral lepton decaying to e j j final state.

To create a sample where the HNL decays to e e nu final state change the following lines in mg5_proc_card.dat

generate e+ e- > ve n1, (n1 > e j j)
add process e+ e- > ve~ n1, (n1 > e j j)

to

generate e+ e- > ve n1, (n1 > e e nu)
add process e+ e- > ve~ n1, (n1 > e e nu)

First create the LHE file. To do this, you'll need to download the latest version of madgraph, and make sure you're using python 3.7 or greater. Then you can do:

```
./bin/mg5_aMC mg5_proc_card.dat
```
to create the LHE file.


The resulting events will be stored in  HNL_ljj/Events/run_01/unweighted_events.lhe.gz file.

Unzip it and give the path to HNL_pythia.cmnd file to generate the delphes root file.

You also need to grab the latest official Delphes card and edm4hep tcl file:
```
#cd to one directory up from FCCeePhysicsPerformance/
git clone https://github.com/HEP-FCC/FCC-config.git
cd FCC-config/
git checkout spring2021
cd ../FCCeePhysicsPerformance/case-studies/BSM/LLP/DisplacedHNL/HNL_sample_creation/
```

To create delphes root file you need to do the following on your command line:

```
source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
DelphesPythia8_EDM4HEP ../../../../../../FCC-config/FCCee/Delphes/card_IDEA.tcl ../../../../../../FCC-config/FCCee/Delphes/edm4hep_IDEA.tcl HNL_pythia.cmnd HNL_ejj.root
```

the resulting HNL_ejj.root is your EDM sample.
