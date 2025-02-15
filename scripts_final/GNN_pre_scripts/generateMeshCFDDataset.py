import os
import copy

import matplotlib.pyplot as plt
import numpy as np
import math

class GenerateMeshOneObject:

    def __init__(self, object, variable_dict_variant,variable_dict_invariant, save_path):
        self.object = object


        self.variable_dict_variant = variable_dict_variant
        self.variable_dict_invariant = variable_dict_invariant
        self.save_path = save_path

        if self.object["type"] == 'airfoil':
            if self.variable_dict_variant["ncyl"] %2 == 1:
                self.variable_dict_variant["ncyl"] += 1

        # print(self.object)
        # print(self.variable_dict_variant)

        self.get_variable_list()
        self.get_objects()
        self.get_boxes()
        self.get_curve_loops()
        self.get_plane_surfaces()

        self.get_transfinite_curves()
        self.get_physical_surfaces()

    def get_airfoil_coordinates(self,NACA_airfoil, n_nodes, chord, alpha):
        M = int(NACA_airfoil[0]) / 100
        P = int(NACA_airfoil[1]) / 10
        T = int(NACA_airfoil[2:]) / 100
        # beta_tab = np.linspace(0, np.pi, math.ceil(n_nodes / 2))
        # print(beta_tab)
        # x_tab = (1 - np.cos(beta_tab)) / 2
        # x_tab = np.linspace(0,1,math.ceil(n_nodes / 2))

        beta_tab = np.linspace(0, np.pi * 9 / 10, math.ceil(n_nodes / 2))
        x_tab = (1 - np.cos(beta_tab)) / 2
        x_tab = x_tab / np.max(x_tab)

        print(x_tab)
        camber_tab = np.zeros(math.ceil(n_nodes / 2))
        gradient_tab = np.zeros(math.ceil(n_nodes / 2))
        thickness_tab = np.zeros(math.ceil(n_nodes / 2))
        x_upper = np.zeros(math.ceil(n_nodes / 2))
        x_lower = np.zeros(math.ceil(n_nodes / 2))
        y_upper = np.zeros(math.ceil(n_nodes / 2))
        y_lower = np.zeros(math.ceil(n_nodes / 2))

        for i in range(len(camber_tab)):
            thickness_tab[i] = (T / 0.2) * ((0.2969 * (x_tab[i] ** (0.5))) +
                                            (-0.126 * (x_tab[i])) +
                                            (-0.3516 * (x_tab[i] ** (2))) +
                                            (0.2843 * (x_tab[i] ** (3))) +
                                            (-0.1036 * (x_tab[i] ** (4))))
            if x_tab[i] < P:
                camber_tab[i] = (M / (P ** 2)) * (2 * P * x_tab[i] - x_tab[i] ** 2)
                gradient_tab[i] = (2 * M / (P ** 2)) * (P - x_tab[i])
            else:
                camber_tab[i] = (M / ((1 - P) ** 2)) * (1 - 2 * P + 2 * P * x_tab[i] - x_tab[i] ** 2)
                gradient_tab[i] = (2 * M / ((1 - P) ** 2)) * (P - x_tab[i])
            x_upper[i] = x_tab[i] - thickness_tab[i] * np.sin(np.arctan(gradient_tab[i]))
            x_lower[i] = x_tab[i] + thickness_tab[i] * np.sin(np.arctan(gradient_tab[i]))
            y_upper[i] = camber_tab[i] + thickness_tab[i] * np.cos(np.arctan(gradient_tab[i]))
            y_lower[i] = camber_tab[i] - thickness_tab[i] * np.cos(np.arctan(gradient_tab[i]))

        x_data = np.append(np.flip(x_lower), x_upper[1:])
        y_data = np.append(np.flip(y_lower), y_upper[1:])

        x_data = x_data * chord
        y_data = y_data * chord

        alpha_rad = np.radians(-alpha)
        cos_alpha = np.cos(alpha_rad)
        sin_alpha = np.sin(alpha_rad)

        x_shifted = x_data - (chord / 2)
        y_shifted = y_data

        x_rotated = cos_alpha * x_shifted - sin_alpha * y_shifted + (chord / 2)
        y_rotated = sin_alpha * x_shifted + cos_alpha * y_shifted

        return x_rotated[:-1], y_rotated[:-1]

    def get_airfoil_list(self,NACA_airfoil, n_nodes, chord,i_airfoil,alpha,x_leadingEdge,y_leadingEdge):
        x,y = self.get_airfoil_coordinates(NACA_airfoil, n_nodes+2, chord,alpha)

        dst = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        print(dst)

        # plt.figure()
        # plt.plot(x,y,"-o")
        # plt.ylim(-0.5,0.5)
        # plt.show()
        x += x_leadingEdge
        y += y_leadingEdge

        script_one_airfoil = [f"// AIRFOIL BEGIN -----------------------------------------\n"]
        i = i_airfoil * 1000 + 1
        for j in range (len(x)):
            script_one_airfoil.append(f"Point({i}) = {{{x[j]}, {y[j]}, 0, 1}};\n")
            script_one_airfoil.append(f"//+\n")
            i += 1
        script_one_airfoil.append(f"\n")

        i = i_airfoil * 1000 + 1
        for j in range (len(x)):
            if j != len(x)-1:
                script_one_airfoil.append(f"Line({i}) = {{{i}, {i+1}}};\n")
            else:
                script_one_airfoil.append(f"Line({i}) = {{{i}, {i_airfoil * 1000 + 1}}};\n")

            script_one_airfoil.append(f"//+\n")
            i += 1

        script_one_airfoil.append(f"\n")

        script_one_airfoil.append("Curve Loop(" + str(i_airfoil) + ") = {" + ", ".join(str(i) for i in range(i_airfoil * 1000 + 1,i_airfoil * 1000 + 1+len(x))) + "};\n")
        script_one_airfoil.append(f"// AIRFOIL END -----------------------------------------\n")
        script_one_airfoil.append(f"\n")

        return script_one_airfoil

    def get_variable_list(self):
        self.script_variable_list = []
        for key in self.variable_dict_variant.keys():
            new_line = key + " = " + str(self.variable_dict_variant[key]) + ";\n"
            self.script_variable_list.append(new_line)

        self.script_variable_list.append(f"\n")

        print(self.script_variable_list)

    def get_objects(self):
        self.script_objects = []
        i = 6
        j = 500

        if self.object["type"] == "circle":
            new_line = f"Circle({i}) = {{{self.object['x']}, {self.object['y']}, 0, {self.object['r']}, 0, 2*Pi}};\n"
            self.script_objects.append(new_line)

            self.box_dict = {"x1": self.object["x"] - 2 * self.object['r'], "x2": self.variable_dict_invariant['x'], "y1": self.object["y"] - 2 * self.object['r'], "y2": self.object["x"] + 2 * self.object['r'], "nx": "nx_box", "ny": "ny_box"}

        elif self.object["type"] == "ellipse":
            new_line = f"Ellipse({i}) = {{{self.object['x']}, {self.object['y']}, 0, {self.object['rx']}, {self.object['ry']}, 0, 2*Pi}};\n"
            self.script_objects.append(new_line)
            new_line = f"Rotate {{{{0, 0, 1}}, {{{self.object['x']}, {self.object['y']}, 0}}, {self.object['alpha'] * -np.pi/180}}} {{Curve{{{i}}};}}\n"
            self.script_objects.append(new_line)

            self.box_dict = {"x1": self.object["x"] - 2 * self.object['rx'], "x2": self.variable_dict_invariant['x'], "y1": self.object["y"] - 2 * self.object['ry'], "y2": self.object["y"] + 2 * self.object['ry'], "nx": "nx_box", "ny": "ny_box"}

        elif self.object["type"] == "rectangle":
            self.object['x1'] = self.object["x"] - 0.5 * self.object["w"]
            self.object['x2'] = self.object["x"] + 0.5 * self.object["w"]
            self.object['y1'] = self.object["y"] - 0.5 * self.object["h"]
            self.object['y2'] = self.object["y"] + 0.5 * self.object["h"]

            tab = self.get_box(self.object,j,i-3,alpha=self.object["alpha"])
            # j+=4
            self.script_objects += tab

            self.box_dict = {"x1": self.object["x"] - self.object['w'], "x2": self.variable_dict_invariant['x'], "y1": self.object["y"] - self.object['h'], "y2": self.object["x"] + self.object['h'], "nx": "nx_box", "ny": "ny_box"}

        elif self.object["type"] == "airfoil":
            self.box_dict = {"x1": self.object["x"] - self.object['c'], "x2": self.variable_dict_invariant['x'], "y1": self.object["y"] - self.object['c'], "y2": self.object["y"] + self.object['c'], "nx": "nx_box", "ny": "ny_box"}


            self.object["x"] -= 0.5 * self.object["c"]
            tab = self.get_airfoil_list(self.object["naca"], self.variable_dict_variant[self.object["n"]], self.object["c"], i - 4, self.object["alpha"], self.object["x"], self.object["y"])
            self.script_objects += tab


            self.script_objects.append(f"//+\n")
            # i += 1

        self.script_objects.append(f"\n")

        # Rectangle(221) = {0.5, 0.9, 0.1, 1, 0.5, 0};

        print(self.script_objects)

    def get_box(self,dict,i,j,alpha=0):
        script_box = []
        x_tab = np.array([dict['x1'],dict['x1'],dict['x2'],dict['x2']])
        y_tab = np.array([dict['y1'],dict['y2'],dict['y2'],dict['y1']])

        x_center = (dict['x1'] + dict['x2']) / 2
        y_center = (dict['y1'] + dict['y2']) / 2

        alpha_rad = np.radians(-alpha)

        cos_alpha = np.cos(alpha_rad)
        sin_alpha = np.sin(alpha_rad)

        x_shifted = x_tab - x_center
        y_shifted = y_tab - y_center

        x_tab = cos_alpha * x_shifted - sin_alpha * y_shifted + x_center
        y_tab = sin_alpha * x_shifted + cos_alpha * y_shifted + y_center

        box_points = [f"// BOX {j - 1 - 1} BEGIN -----------------------------------------\n",
                      f"Point({i + 1}) = {{{x_tab[0]}, {y_tab[0]}, 0}};\n",
                      f"//+\n",
                      f"Point({i + 2}) = {{{x_tab[1]}, {y_tab[1]}, 0}};\n",
                      f"//+\n",
                      f"Point({i + 3}) = {{{x_tab[2]}, {y_tab[2]}, 0}};\n",
                      f"//+\n",
                      f"Point({i + 4}) = {{{x_tab[3]}, {y_tab[3]}, 0}};\n",
                      f"//+\n",
                      f"\n"]

        box_lines = [f"Line({i + 1}) = {{{i + 1}, {i + 2}}};\n",
                     f"//+\n",
                     f"Line({i + 2}) = {{{i + 2}, {i + 3}}};\n",
                     f"//+\n",
                     f"Line({i + 3}) = {{{i + 3}, {i + 4}}};\n",
                     f"//+\n",
                     f"Line({i + 4}) = {{{i + 4}, {i + 1}}};\n",
                     f"//+\n",
                     f"\n"]

        curve_loop = ["Curve Loop(" + str(j) + ") = {" + ", ".join(str(i) for i in range(i + 1, 5 + i)) + "};\n",
                      f"\n"]

        transfinite_curves = [
            f'Transfinite Curve {{{i + 1}}} = {dict["ny"]} Using Progression 1;\n',
            f"//+\n",
            f'Transfinite Curve {{{i + 2}}} = {dict["nx"]} Using Progression 1;\n',
            f"//+\n",
            f'Transfinite Curve {{{i + 3}}} = {dict["ny"]} Using Progression 1;\n',
            f"//+\n",
            f'Transfinite Curve {{{i + 4}}} = {dict["nx"]} Using Progression 1;\n',
            f"//+\n",
            f"// BOX {j - 1 - 1} END -------------------------------------------\n", ]

        script_box += box_points
        script_box += box_lines
        script_box += curve_loop
        script_box += transfinite_curves

        return script_box

    def get_boxes(self):
        self.script_boxes = []
        j = 3
        i = 100


        self.script_boxes += self.get_box(self.box_dict,i,j)

            # ["Plane Surface(1) = {" + ", ".join(str(i) for i in plane_list_main) + "};\n"]

        self.script_boxes.append(f"\n")


        print(self.script_boxes)

    def get_curve_loops(self):
        self.script_curve_loops = [f"Curve Loop(1) = {{1, 2, 3, -102, -101, -104, 4, 5}};\n",
                                   f"//+\n"]
        i = 1
        if self.object["type"] in ["circle","ellipse"]:
            new_line = "Curve Loop(" + str(i + 1) + ") = {" + str(i + 5) + "};\n"
            self.script_curve_loops.append(new_line)
            self.script_curve_loops.append(f"//+\n")
        self.script_curve_loops.append(f"\n")

        print(self.script_curve_loops)

    def get_plane_surfaces(self):
        plane_list_main = [1]
        i = 3

        extra_planes_list = []
        sub_planes_list = [i] + [2]
        extra_planes_list.append(sub_planes_list)

        # self.script_plane_surfaces = ["Plane Surface(1) = {" + ", ".join(str(i) for i in range(1, len(objects_list) + 2)) + "};\n"]
        self.script_plane_surfaces = ["Plane Surface(1) = {" + ", ".join(str(i) for i in plane_list_main) + "};\n"]
        self.script_plane_surfaces.append(f"//+\n")

        i = 2
        for tab in extra_planes_list:
            new_line = f"Plane Surface({i}) = {{" + ", ".join(str(i) for i in tab) + "};\n"
            i += 1
            self.script_plane_surfaces.append(new_line)
            self.script_plane_surfaces.append(f"//+\n")

        self.script_plane_surfaces.append(f"\n")


        print(self.script_plane_surfaces)

    def get_transfinite_curves(self):
        # type_list = ["Bump", "Progression", "Progression", "Progression"]
        # value_list = [0.2,1,0.8,1]
        #
        # type_list = ["Progression", "Progression", "Progression", "Progression"]
        # value_list = [1,1,1,1]
        self.script_transfinite_curves = [f'Transfinite Curve {{1}} = ny Using Progression 1;\n',
                        f"//+\n",
                        f'Transfinite Curve {{2}} = nx Using Progression 1;\n',
                        f"//+\n",
                        f'Transfinite Curve {{3}} = ny/2 Using Progression 1;\n',
                        f"//+\n",
                        f'Transfinite Curve {{4}} = ny/2 Using Progression 1;\n',
                        f"//+\n",
                        f'Transfinite Curve {{5}} = nx Using Progression 1;\n',
                        f"//+\n"]

        new_line = "Transfinite Curve {" + str(6) + "} = " + str(
            self.object["n"]) + " Using Progression 1;\n"

        self.script_transfinite_curves.append(new_line)
        self.script_transfinite_curves.append(f"//+\n")

        self.script_transfinite_curves.append(f"\n")

    def get_physical_surfaces(self):
        side_index_list = [6 + 1 + 4 * 1]
        object_index_list = np.arange(6,6+1)

        object_index_list[0] = 1
        side_index_list.append(side_index_list[-1] + 2)

            # side_index_list.append(side_index_list[-1]+i)
        if self.object["type"] in ["circle","ellipse"]:
            i_box = 14
        else:
            i_box = 13 + self.variable_dict_variant[self.object["n"]]

        self.script_physical_surfaces = [f'Physical Volume("Fluid", 1) = {{{", ".join(str(i) for i in range(1, 2 + 1))}}};\n',
                        f"//+\n",
                        f'Physical Surface("Left", 2) = {{{3}}};\n',
                        f"//+\n",
                        f'Physical Surface("Lower", 3) = {{10}};\n',
                        f"//+\n",
                        f'Physical Surface("Right", 4) = {{5,12,9}};\n',
                        f"//+\n",
                        f'Physical Surface("Upper", 5) = {{4}};\n',
                        f"//+\n",
                        f'Physical Surface("Side", 6) = {{1, 2, 11, {i_box}}};\n',
                        f"//+\n"]

        if self.object["type"] in ["circle","ellipse"]:
            new_line = f'Physical Surface("Object", {7}) = {{{13}}};\n'

        else:
            new_line = f'Physical Surface("Object", {7}) = {{{", ".join(str(i) for i in range(13, i_box))}}};\n'

        self.script_physical_surfaces.append(new_line)
        self.script_physical_surfaces.append(f"//+\n")

        print(self.script_physical_surfaces)

    def get_mesh(self):

        script_start = [f"// Gmsh project created on Sun May 19 18:32:31 2024\n",
                        f'SetFactory("OpenCASCADE");\n',
                        f"\n"]

        script_points = [f"Point(1) = {{0, 0, 0}};\n",
                        f"//+\n",
                        f"Point(2) = {{0, {self.variable_dict_invariant['y']}, 0}};\n",
                        f"//+\n",
                        f"Point(3) = {{{self.variable_dict_invariant['x']}, {self.variable_dict_invariant['y']}, 0}};\n",
                        f"//+\n",
                        f"Point(4) = {{{self.variable_dict_invariant['x']}, 0, 0}};\n",
                        f"//+\n",
                        f"\n"]

        script_lines = [f"Line(1) = {{1, 2}};\n",
                        f"//+\n",
                        f"Line(2) = {{2, 3}};\n",
                        f"//+\n",
                        f"Line(3) = {{3, 103}};\n",
                        f"//+\n",
                        f"Line(4) = {{104, 4}};\n",
                        f"//+\n",
                        f"Line(5) = {{4, 1}};\n",
                        f"//+\n",
                        f"\n"]

        extrude_list = [f"Extrude {{0, 0, 1}} {{\n",
                        f'Surface{{{", ".join(str(i) for i in np.arange(1,3))}}}; Layers {{1}}; Recombine;\n',
                        f"}}\n",
                        f"//+\n",
                        f"\n"]

        end_list = [f"Mesh.Algorithm = 1;\n"]

        script = (script_start + self.script_variable_list + self.script_boxes +
                  script_points + script_lines + self.script_objects +
                  self.script_curve_loops + self.script_plane_surfaces +
                  self.script_transfinite_curves + extrude_list +
                  self.script_physical_surfaces + end_list)
        print(script)

        with open(self.save_path, 'w') as new_file:
            new_file.writelines(script)



if __name__ == '__main__':

    variable_dict_variant = {"nx": 24,
                             "ny": 24,
                            "ncyl": 200,
                            "nx_box": 150,
                            "ny_box": 25,
                            "nx_rect": 50,
                            "ny_rect": 50,
                             }

    variable_dict_variant_base = {"nx": 32,
                                  "ny": 32,
                                  "ncyl": 200,
                                  "nx_box": 128,
                                  "ny_box": 16,
                                  }
    factor = 323/200
    factor = 1


    variable_dict_variant = {"nx": int(32*factor),
                                  "ny": int(32*factor),
                                  "ncyl": int(212*factor),
                                  "nx_box": int(128*factor),
                                  "ny_box": int(16*factor),
                                  }

    variable_dict_invariant = {
                            "y": 16,
                            "x": 28,
                     }
    #
    # variable_dict_variant = {"nx": 32,
    #                               "ny": 32,
    #                               "ncyl": 11,
    #                               "nx_box": 160,
    #                               "ny_box": 20,
    #                               }

    for key in variable_dict_variant.keys():
        variable_dict_variant[key] = int(round(variable_dict_variant[key]*1))

    # object = {"type":"circle", "x":8, "y": 8, "r": 0.5, "n":"ncyl"}
    object = {"type":"airfoil","naca": "2412", "x":8, "y": 8, "c": 1,"alpha":0, "n":"ncyl","box": True}


    save_path = "/home/justinbrusche/dataVaryingMeshes/geo_file_random.geo"

    a = GenerateMeshOneObject(object, variable_dict_variant,variable_dict_invariant, save_path)
    a.get_mesh()

    # print(f"Point(1) = {{{a}, {b}, {c}}};\n")


