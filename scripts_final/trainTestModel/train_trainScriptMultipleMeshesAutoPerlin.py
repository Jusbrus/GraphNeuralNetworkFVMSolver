import os
import numpy as np
import yaml
import sys
import pickle
import torch
import torch.optim as optim
import time

from Utilities.plotScripts import *
from Utilities._loadModel import *
from Utilities.embed_data import *
from Utilities.LossFunctionScriptMeshes import *
config_file_list = ['config_files/foundParam_conv_4666_e3_100_small.yaml']



def run(config_file):
    countCellweights = False
    save_every_model = True
    device = 'cuda'
    normalize_data = True
    compute_diff_cfd = False

    nSets = 21
    nPerCase = 1000
    testRatio = 0.1
    if nSets == 1:
        train_cases = np.array([0])
    else:
        test_cases = [11,12]
        # test_cases = [5,15]

        train_cases = np.delete(np.arange(nSets), test_cases)

    with open(config_file, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    test_case_path = conf['test_case_path']

    dataDictMeshesList = []
    trainDataLoaderList = []
    testDataLoaderList = []
    GNNdataDictList = []

    nSetsTrain = len(train_cases)

    if nSets == 1:
        iTab = np.arange(nPerCase)
        iTabTest = iTab[::10]

        iTabTrain = np.delete(iTab, iTabTest)
        nPerSetTrain = len(iTabTrain)
        nPerSetTest = len(iTabTest)

    else:
        iTabTrain = np.arange(int(round((1 - testRatio) * nPerCase)))
        iTabTest = np.arange(int(round((1 - testRatio) * nPerCase)), nPerCase)
        nPerSetTrain = len(iTabTrain)
        nPerSetTest = len(iTabTest)

    for i in train_cases:
        casePath = os.path.join(test_case_path, f"case_{i}")

        file_path = f"{casePath}/dataDictMeshes.pkl"
        with open(file_path, 'rb') as file:
            dataDictMeshesList.append(pickle.load(file))

        trainDataLoaderList.append(loadDataLoader(casePath, iTabTrain, conf['batchSize'], dataDictMeshesList[-1],
                                                  BCInput=conf["modelParam"]["inputChannels"]["BCInput"],
                                                  gradientBCInput=conf["modelParam"]["inputChannels"]["gradientBCInput"], doShuffling=True,
                                                  normalize_data=normalize_data, compute_diff_cfd=compute_diff_cfd))
        testDataLoaderList.append(loadDataLoader(casePath, iTabTest,
                                                 conf['batchSize'], dataDictMeshesList[-1],
                                                 BCInput=conf["modelParam"]["inputChannels"]["BCInput"],
                                                 gradientBCInput=conf["modelParam"]["inputChannels"]["gradientBCInput"],
                                                 doShuffling=False, normalize_data=normalize_data, compute_diff_cfd=compute_diff_cfd))

        file_path = f"{casePath}/embedding_{conf['edgeAttrName']}"
        with open(file_path, 'rb') as file:
            GNNdataDictList.append(pickle.load(file))

    # evaluateDataDictMeshesList = []
    # evaluateDataLoaderList = []
    # for i in test_cases:
    #     casePath = os.path.join(test_case_path, f"case_{i}")
    #
    #     file_path = f"{casePath}/dataDictMeshes.pkl"
    #     with open(file_path, 'rb') as file:
    #         evaluateDataDictMeshesList.append(pickle.load(file))
    #
    #     trainDataLoaderList.append(loadDataLoader(casePath, np.arange(nPerCase), conf['batchSize'], trainDataDictMeshesList[-1],
    #                                      BCInput=conf["modelParam"]["inputChannels"]["BCInput"],
    #                                      gradientBCInput=conf["modelParam"]["inputChannels"]["gradientBCInput"],doShuffling=True,
    #                                      normalize_data=normalize_data, compute_diff_cfd=compute_diff_cfd))


    # Initialize model
    sys.path.append("/home/justinbrusche/scripts_final/trainTestModel")
    model = loadModel(conf,GNNdataDictList[0]["GNNData"],device)

    # Load pretrained parameters
    model_path = os.path.join(conf["modelDir"], f'{conf["modelFilename"]}.pth')
    print(model_path)
    if os.path.exists(model_path):
        print("Pre-trained model found. Loading model.")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        optimizer = optim.Adam(model.parameters(), lr=conf["modelParam"]["lr"])
        if config_file != 'config_files/cfd_mesh_conv_4666_e3_100_small.yaml':
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resume = True
    else:
        print("No pre-trained model found. Initializing new model.")
        start_epoch = 0
        optimizer = optim.Adam(model.parameters(), lr=conf["modelParam"]["lr"])
        resume = False

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of learnable parameters in the model: {num_params}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=10, threshold=3e-4,
                                                     threshold_mode='rel')

    lossFunction = LossClass(test_case_path,train_cases, device)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    # Function to run an epoch
    def runEpoch(trainDataLoaderList, testDataLoaderList, model, optimizer, scheduler, lossFunction, epoch,dataDictMeshesList,GNNdataDictList,nSetsTrain,nPerSetTrain,nPerSetTest):
        source_factor = 0
        fvm_factor = 0

        # print(train_cases)

        i_train_cases_shuffled = np.random.permutation(np.arange(nSetsTrain))
        # print(i_train_cases_shuffled)

        model.train()
        t = time.time()
        pressure_loss_list = []
        train_loss = 0
        lossSource_sum = 0
        lossPressure_sum = 0
        lossFVM_sum = 0
        first_outP = None  # To store the first output

        # for i, (pEqnD, pEqnFace, pEqnSource) in enumerate(trainDataLoader):
        for i in range(nSetsTrain):
            iSet = i_train_cases_shuffled[i]
            GNNdataDict = GNNdataDictList[iSet]
            dataDictMeshes = dataDictMeshesList[iSet]
            save_data = True
            for p, pEqnD, pEqnFace, pEqnSource, inputCellArray, faceArray_incl_boundaries, BCArray, *factors in trainDataLoaderList[iSet]:

                if len(factors) == 0:
                    factors = [torch.tensor(1),torch.tensor(1),torch.tensor(1),torch.tensor(1)]

                optimizer.zero_grad()

                if compute_diff_cfd:
                    inputFaceArray = torch.zeros((len(p), dataDictMeshes[0]["Nfaces"], 2))
                    inputFaceArray[:, dataDictMeshes[0]["boundaryFaces"], :] = BCArray
                    inputFaceArray = torch.cat((faceArray_incl_boundaries, inputFaceArray), dim=2)

                elif conf["modelParam"]["inputChannels"]["BCInput"] == True:
                    if conf["modelParam"]["inputChannels"]["gradientBCInput"] == True:
                        inputFaceArray = torch.zeros((len(p), dataDictMeshes[0]["Nfaces"],4))
                        inputFaceArray[:,dataDictMeshes[0]["boundaryFaces"],:] = BCArray

                    else:
                        inputFaceArray = torch.zeros((len(p), dataDictMeshes[0]["Nfaces"],3))

                        inputFaceArray[:,dataDictMeshes[0]["boundaryFaces"],:] = BCArray

                    inputFaceArray = torch.cat((faceArray_incl_boundaries,inputFaceArray), dim=2)
                else:
                    inputFaceArray = faceArray_incl_boundaries+0

                outP = model(inputCellArray.to(device), inputFaceArray.to(device), GNNdataDict["graphData"], GNNdataDict["graphPoolData"],
                             GNNdataDict["GNNDataForward"], GNNdataDict["poolDataForward"])

                lossPressure = lossFunction.computeLossPressure(outP, p.to(device))
                #lossSource = lossFunction.computeLossSource(outP,p.unsqueeze(-1).to(device), pEqnD.to(device), pEqnFace.to(device), pEqnSource.to(device),factors[3],iSet)
                #lossFVM = lossFunction.computeLossFVM(outP, p.to(device), pEqnFace.to(device),iSet)

                pressure_loss_list.append(lossPressure.item())

                lossPressure_sum += lossPressure.item()
                #lossSource_sum += source_factor * lossSource.item()
                #lossFVM_sum += fvm_factor * lossFVM.item()

                # loss = source_factor * lossSource + lossPressure + fvm_factor * lossFVM
                loss = lossPressure +0
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                save_data = False

                if save_data:
                    if first_outP is None:
                        first_outP = [outP[:5].detach().cpu().numpy()]  # Save the first 5 elements of the first batch
                        first_p = [p[:5].detach().cpu().numpy()]
                        first_pEqnSource = [pEqnSource[:5].detach().cpu().numpy()]

                    else:
                        first_outP = first_outP + [outP[:5].detach().cpu().numpy()]
                        first_p = first_p + [p[:5].detach().cpu().numpy()]
                        first_pEqnSource = first_pEqnSource + [pEqnSource[:5].detach().cpu().numpy()]

                    save_data = False

            if i%3==0:
                # print(i,train_loss/(nPerSetTrain*(i+1)),lossPressure_sum/(nPerSetTrain*(i+1)),lossSource_sum/(nPerSetTrain*(i+1)),lossFVM_sum/(nPerSetTrain*(i+1)))
                print(i,train_loss/(nPerSetTrain*(i+1)),lossPressure_sum/(nPerSetTrain*(i+1)))


        avg_train_loss = train_loss / (nPerSetTrain * nSetsTrain)

        first_outP_test = None  # To store the first output

        loss_list = []

        # Evaluate on test data

        model.eval()
        test_loss = 0
        lossSource_sum = 0
        lossPressure_sum = 0
        lossFVM_sum = 0
        with torch.no_grad():
            t1 = time.time()
            for i in range(nSetsTrain):
                iSet = i_train_cases_shuffled[i]
                GNNdataDict = GNNdataDictList[iSet]
                dataDictMeshes = dataDictMeshesList[iSet]

                save_data = True
                for p, pEqnD, pEqnFace, pEqnSource, inputCellArray, faceArray_incl_boundaries, BCArray, *factors  in testDataLoaderList[iSet]:

                    if len(factors) == 0:
                        factors = [torch.tensor(1), torch.tensor(1), torch.tensor(1), torch.tensor(1)]

                    if compute_diff_cfd:
                        inputFaceArray = torch.zeros((len(p), dataDictMeshes[0]["Nfaces"], 2))
                        inputFaceArray[:, dataDictMeshes[0]["boundaryFaces"], :] = BCArray
                        inputFaceArray = torch.cat((faceArray_incl_boundaries, inputFaceArray), dim=2)

                    elif conf["modelParam"]["inputChannels"]["BCInput"] == True:
                        if conf["modelParam"]["inputChannels"]["gradientBCInput"] == True:
                            inputFaceArray = torch.zeros((len(p), dataDictMeshes[0]["Nfaces"], 4))
                            inputFaceArray[:, dataDictMeshes[0]["boundaryFaces"], :] = BCArray

                        else:
                            inputFaceArray = torch.zeros((len(p), dataDictMeshes[0]["Nfaces"], 3))

                            inputFaceArray[:, dataDictMeshes[0]["boundaryFaces"], :] = BCArray

                        inputFaceArray = torch.cat((faceArray_incl_boundaries, inputFaceArray), dim=2)
                    else:
                        inputFaceArray = faceArray_incl_boundaries + 0

                    outP = model(inputCellArray.to(device), inputFaceArray.to(device), GNNdataDict["graphData"],
                                 GNNdataDict["graphPoolData"],
                                 GNNdataDict["GNNDataForward"], GNNdataDict["poolDataForward"])

                    lossPressure = lossFunction.computeLossPressure(outP, p.to(device))
                    # lossSource = lossFunction.computeLossSource(outP, p.unsqueeze(-1).to(device), pEqnD.to(device),
                    #                                             pEqnFace.to(device), pEqnSource.to(device), factors[3],
                    #                                             iSet)
                    # lossFVM = lossFunction.computeLossFVM(outP, p.to(device), pEqnFace.to(device), iSet)

                    lossPressure_sum += lossPressure.item()
                    # lossSource_sum += source_factor * lossSource.item()
                    # lossFVM_sum += fvm_factor * lossFVM.item()

                    # loss = source_factor * lossSource + lossPressure + fvm_factor * lossFVM
                    loss = lossPressure + 0

                    test_loss += loss.item()
                    save_data = False
                    if save_data:
                        if first_outP_test is None:
                            first_outP_test = [outP[:5].detach().cpu().numpy()]  # Save the first 5 elements of the first batch
                            first_p_test = [p[:5].detach().cpu().numpy()]
                            first_pEqnSource_test = [pEqnSource[:5].detach().cpu().numpy()]

                        else:
                            first_outP_test = first_outP_test + [outP[:5].detach().cpu().numpy()]
                            first_p_test = first_p_test + [p[:5].detach().cpu().numpy()]
                            first_pEqnSource_test = first_pEqnSource_test + [pEqnSource[:5].detach().cpu().numpy()]

                        save_data = False

                # print(i, test_loss / (nPerSetTest * (i + 1)), lossPressure_sum / (nPerSetTest * (i + 1)),
                #       lossSource_sum / (nPerSetTest * (i + 1)), lossFVM_sum / (nPerSetTest * (i + 1)))


            print("total test time: ", time.time() - t1, ", avg time per field: ", (time.time() - t1) / (nPerSetTest * nSetsTrain))

        avg_test_loss = test_loss / (nPerSetTest * nSetsTrain)
        print(f'Epoch {epoch} - Train Loss: {avg_train_loss:.4f} - Time: {time.time() - t:.2f}s - Test Loss: {avg_test_loss:.4f}')
        # Step the scheduler

        current_lr = get_lr(optimizer)

        scheduler.step(avg_test_loss)

        # Get the updated learning rate after the scheduler step
        new_lr = get_lr(optimizer)

        # Check if the learning rate has changed
        print(f'Learning rate: {new_lr}')
        if new_lr != current_lr:
            print(f"Epoch {epoch + 1}: Learning rate adjusted from {current_lr:.6f} to {new_lr:.6f}")

        # plt.plot(loss_list)
        # plt.show()

        return avg_train_loss, avg_test_loss

    # Run multiple epochs and save the model intermittently
    num_epochs = conf['maxEpochs']
    save_interval = conf['freqToFile']

    log_file_path = os.path.join(conf["modelDir"], 'training_log.pkl')

    # Check if the log file already exists
    if os.path.exists(log_file_path):
        with open(log_file_path, 'rb') as log_file:
            log_dict = pickle.load(log_file)
    else:
        # Initialize the log dictionary if it doesn't exist
        log_dict = {"Epoch": [], "Train Loss": [], "Test Loss": []}

    datadictSaveMesh_path = os.path.join(conf["modelDir"], f'dataDictMeshesList.pkl')
    with open(datadictSaveMesh_path, 'wb') as f:
        pickle.dump(dataDictMeshesList, f)

    for epoch in range(start_epoch, num_epochs):
        train_loss, test_loss = runEpoch(trainDataLoaderList, testDataLoaderList, model, optimizer, scheduler,
                                                                                                                                    lossFunction, epoch, dataDictMeshesList,GNNdataDictList,nSetsTrain,nPerSetTrain,nPerSetTest)
        # Append the losses to the log dictionary
        log_dict["Epoch"].append(epoch + 1)
        log_dict["Train Loss"].append(train_loss)
        log_dict["Test Loss"].append(test_loss)

        # Save the log dictionary after each epoch
        with open(log_file_path, 'wb') as log_file:
            pickle.dump(log_dict, log_file)

        # Save the model at specified intervals
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
            save_path = os.path.join(conf["modelDir"], f'{conf["modelFilename"]}.pth')
            save_path_epoch = os.path.join(conf["modelDir"], f'{conf["modelFilename"]}_{epoch+1}.pth')

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss
            }, save_path)
            if save_every_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss
                }, save_path_epoch)

            print(f'Model saved at epoch {epoch + 1}')
            # Save the first 5 elements of outP
            # outP_save_path = os.path.join(conf["modelDir"], f'outP_epoch_{epoch + 1}.pkl')
            #
            # with open(outP_save_path, 'wb') as f:
            #     pickle.dump(first_outP, f)
            #
            # p_save_path = os.path.join(conf["modelDir"], f'p_epoch_{epoch + 1}.pkl')
            # with open(p_save_path, 'wb') as f:
            #     pickle.dump(first_p, f)
            #
            # pEqnSource_save_path = os.path.join(conf["modelDir"], f'pEqnSource_epoch_{epoch + 1}.pkl')
            # with open(pEqnSource_save_path, 'wb') as f:
            #     pickle.dump(first_pEqnSource, f)
            #
            # # Save test data
            # outP_save_path = os.path.join(conf["modelDir"], f'test_outP_epoch_{epoch + 1}.pkl')
            # with open(outP_save_path, 'wb') as f:
            #     pickle.dump(first_outP_test, f)
            #
            # p_save_path = os.path.join(conf["modelDir"], f'test_p_epoch_{epoch + 1}.pkl')
            # with open(p_save_path, 'wb') as f:
            #     pickle.dump(first_p_test, f)
            #
            # pEqnSource_save_path = os.path.join(conf["modelDir"], f'test_pEqnSource_epoch_{epoch + 1}.pkl')
            # with open(pEqnSource_save_path, 'wb') as f:
            #     pickle.dump(first_pEqnSource_test, f)
            #
            # i_train_cases_shuffled_path = os.path.join(conf["modelDir"], f'i_train_cases_shuffled_epoch_{epoch + 1}.pkl')
            # with open(i_train_cases_shuffled_path, 'wb') as f:
            #     pickle.dump(i_train_cases_shuffled, f)

            print(f'First 5 elements of outP saved at epoch {epoch + 1}')

    print("Training complete.")


for config_file in config_file_list:
    run(config_file)
