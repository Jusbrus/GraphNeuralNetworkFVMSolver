import pickle


def loadModel(conf,nAttributesDict,device):
    print(nAttributesDict)

    if conf["modelParam"]["model"] == "final_design":
        # from Models.GraphUnetSimple.GraphUnet import *
        from Models.final_design.MeshGraphNetCNN import GraphUNetMediumLessLayers,GraphUNetSmallLessLayers,GraphUNetVerySmallLessLayers,GraphUNetVeryVerySmallLessLayers

        if conf["modelParam"]["size"] == "MediumLessLayers":
            model = GraphUNetMediumLessLayers(nAttributesDict).to(device)
        if conf["modelParam"]["size"] == "SmallLessLayers":
            model = GraphUNetSmallLessLayers(nAttributesDict).to(device)
        if conf["modelParam"]["size"] == "VerySmallLessLayers":
            model = GraphUNetVerySmallLessLayers(nAttributesDict).to(device)
        if conf["modelParam"]["size"] == "VeryVerySmallLessLayers":
            model = GraphUNetVeryVerySmallLessLayers(nAttributesDict).to(device)

    elif conf["modelParam"]["model"] == "final_design_cfd":
        # from Models.GraphUnetSimple.GraphUnet import *
        from Models.final_design_cfd.MeshGraphNetCNN import GraphUNetVeryVeryVerySmallLessLayers,GraphUNetFullScaleLessLayers, GraphUNetMediumLessLayers,GraphUNetSmallLessLayers,GraphUNetVerySmallLessLayers,GraphUNetVeryVerySmallLessLayers

        if conf["modelParam"]["size"] == "FullScaleLessLayers":
            model = GraphUNetFullScaleLessLayers(nAttributesDict).to(device)
        if conf["modelParam"]["size"] == "MediumLessLayers":
            model = GraphUNetMediumLessLayers(nAttributesDict).to(device)
        if conf["modelParam"]["size"] == "SmallLessLayers":
            model = GraphUNetSmallLessLayers(nAttributesDict).to(device)
        if conf["modelParam"]["size"] == "VerySmallLessLayers":
            model = GraphUNetVerySmallLessLayers(nAttributesDict).to(device)
        if conf["modelParam"]["size"] == "VeryVerySmallLessLayers":
            model = GraphUNetVeryVerySmallLessLayers(nAttributesDict).to(device)
        if conf["modelParam"]["size"] == "VeryVeryVerySmallLessLayers":
            model = GraphUNetVeryVeryVerySmallLessLayers(nAttributesDict).to(device)

    if conf["modelParam"]["model"] == "magBase":
        # from Models.GraphUnetSimple.GraphUnet import *
        from Models.magBase.MeshGraphNetCNN import Small, VerySmall, VeryVerySmall, VeryVeryVerySmall

        # model = GraphUNetSimple(GNNdataDict["GNNData"]).to(device)
        if conf["modelParam"]["size"] == "Small":
            model = Small(nAttributesDict).to(device)
        elif conf["modelParam"]["size"] == "VerySmall":
            model = VerySmall(nAttributesDict).to(device)
        elif conf["modelParam"]["size"] == "VeryVerySmall":
            model = VeryVerySmall(nAttributesDict).to(device)
        elif conf["modelParam"]["size"] == "VeryVeryVerySmall":
            model = VeryVeryVerySmall(nAttributesDict).to(device)

    return model







