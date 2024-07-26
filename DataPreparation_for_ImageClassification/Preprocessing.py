import os
import shutil
import random
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt


def set_seed(seed= 42):
    """
    src: https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # # When running on the CuDNN backend, two further options must be set
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # # Set a fixed value for the hash seed
    # os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    return

def save_model(epoch, model_state_dict, optimizer_state_dict, loss, path):
    torch.save(
        {
            'epoch': epoch+1,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'loss': loss,
        }, path)

def save_model_config(cmp_dir, counts, files, train_losses, val_losses, config, report, auroc):
    out = {}
    for cmp in files:
        out[cmp] = (counts[cmp],files[cmp])
    with open(cmp_dir+"/dataset.pickle", 'wb') as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.savefig(cmp_dir+"/plots.png")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    component = cmp_dir.split("/")[-1]
    with open(cmp_dir+"/"+component+"_model_results.txt", "w") as f:
        f.write(component + " report:\n")
        f.write(report)
        f.write("\n"*2)
        f.write("Auroc: "+str(auroc)+"\n")
        f.write("\n"*2)
        f.write("Training data distribution:\n")
        for key, value in counts.items():
            f.write(key+": "+str(value)+"\n")
        f.write("\n"*2)
        f.write("Configs:\n")
        for key, value in config.items():
            f.write(key+": "+ str(value)+"\n")
    
    

class DataCollector(object):
    """
    Class for the aggregation and creation of a model directory under "3_Hardware_Feature_Classification/Classifiers/"
    1. Instantiation asks for the component and the class distribution for the dataset
    2. After Instantiation the member of the false class can be configured through set_neg_categories()
       (By default it will consider every other component except CamBack, CamFront and CamTop)
    3. .create_dataset() will create the directory including the data folder
       - The 0 folder will contain (true_samples/class_dist) samples but no more than 8781 (total # of cropped imgs)
       - in favor for training the conv classifier, it incorporates the best possible variety into the false category
         by balancing the other components.
    """
    #Pfad für cropped images hinterlegen
    component_dir = "Data/Features_Cropped/"
    # In diesem Pfad werden Ornder mit neuen Datensätzen gespeichert
    classifier_dir = "Data/binary_classification/"
    #excl_categories = ["0_CamFront_cropped"] #, "CamFront", "CamTop"]

    def __init__(self, component, class_dist=None, subfolder=""):
        self.component = component
        self.class_dist = class_dist
        self.neg_categories = self.read_categories()
        self.dir = None
        self.total_pos_cat = None
        self.total_neg_cat = None
        self.dist_dict = None
        self.dist_dict_files = None
        self.subfolder = subfolder
        if subfolder != "":
            if not os.path.exists(DataCollector.classifier_dir+self.subfolder):
                os.mkdir(DataCollector.classifier_dir+self.subfolder)

    # Remove unwanted folders (images of cropped boxes)
    def read_categories(self):
        neg_categories = os.listdir(DataCollector.component_dir)
        neg_categories.remove(self.component)
        #for i in DataCollector.excl_categories:
        #    neg_categories.remove(i)
        return neg_categories

    def set_neg_categories(self, new_list):
        self.neg_categories = new_list

    def create_dataset(self):
        # Create new dir in Classifiers
        new_dir = self.unique_dir()
        self.dir = DataCollector.classifier_dir+self.subfolder+new_dir
        os.mkdir(self.dir)
        #os.mkdir(self.dir + "/data")

        # Create data/1 folder for true samples
        shutil.copytree(DataCollector.component_dir + self.component, self.dir + "/1")
        # os.rename(self.dir + "/data/" + self.component, self.dir + "/data/1")
        self.total_pos_cat = len(os.listdir(self.dir + "/1"))

        # Aggregate samples from neg_categories
        os.mkdir(self.dir + "/0")
        self.dist_dict = self.calc_dist()
        self.dist_dict_files = {}
        for cmp, count in self.dist_dict.items():
            img_list = os.listdir(DataCollector.component_dir + cmp)
            random.seed(42)
            random.shuffle(img_list)
            self.dist_dict_files[cmp] = img_list
            for img in img_list[:count]:
                source = DataCollector.component_dir + "/" + cmp + "/" + img
                destination = self.dir + "/0/" + img
                if os.path.isfile(source):
                    shutil.copy(source, destination)
        return self.dir

    def calc_dist(self):
        no_selection = {}
        for cmp in self.neg_categories:
            no_selection[cmp] = len(os.listdir(DataCollector.component_dir + cmp))
        maximum = sum([no_selection[cmp] for cmp in no_selection])
        if self.class_dist is None:
            self.total_neg_cat = maximum
            return no_selection

        selected = 0
        selection = {}
        self.total_neg_cat = round(self.total_pos_cat/self.class_dist-self.total_pos_cat)
        optimal = round(self.total_neg_cat / len(self.neg_categories))
        # in case of rounding up, optimal gets too big
        optimal = optimal -1

        for cmp in self.neg_categories:
            cmp_count = no_selection[cmp]
            if cmp_count <= optimal:
                selection[cmp] = {"count":cmp_count, "complete":True}
                selected += cmp_count
            else:
                selection[cmp] = {"count":optimal, "complete":False}
                selected += optimal

        while selected < min(self.total_neg_cat, maximum):
            for cmp in selection:
                if selection[cmp]["complete"]:
                    continue
                selection[cmp]["count"] += 1
                selected += 1
                if selection[cmp]["count"] == no_selection[cmp]:
                    selection[cmp]["complete"] = True
                if selected == min(self.total_neg_cat, maximum):
                    break

        selection = {cmp:value["count"] for (cmp,value) in selection.items()}
        return selection

    def unique_dir(self):
        dirs = os.listdir(DataCollector.classifier_dir+self.subfolder)
        i = 0
        dir_name = self.component + "_" + str(i)
        while dir_name in dirs:
            i+=1
            dir_name = self.component + "_" + str(i)
        return dir_name

    def get_class_dist(self):
        with_pos = self.dist_dict
        with_pos[self.component] = self.total_pos_cat
        return with_pos
    def get_file_lists(self):
        with_pos = self.dist_dict_files
        with_pos[self.component] = os.listdir(DataCollector.component_dir + self.component)
        return with_pos
