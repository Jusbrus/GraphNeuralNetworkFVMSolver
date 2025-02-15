#include "fvCFD.H"
#include <unordered_set>

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    // Get the cell centers
    const volVectorField& cellCenters = mesh.C();

    // Define the output file path for cell centers
    fileName cellCenterFilePath(runTime.path()/"constant"/"polyMesh"/"cellCenters.txt");

    // Open a file to write cell centers
    OFstream cellCenterFile(cellCenterFilePath);

    for (label cellI = 0; cellI < mesh.nCells(); ++cellI)
    {
        const vector& cellCenter = cellCenters[cellI];
        cellCenterFile << cellCenter.x() << " " << cellCenter.y() << " " << cellCenter.z() << "\n";
    }

    const polyBoundaryMesh& boundaryMesh = mesh.boundaryMesh();

    // Define the output file path for cell faces
    fileName cellFacesFilePath(runTime.path()/"constant"/"polyMesh"/"cellFaces.txt");

    // Open a file to write cell faces
    OFstream cellFacesFile(cellFacesFilePath);

    // Define the output file path for cell points
    fileName cellPointsFilePath(runTime.path()/"constant"/"polyMesh"/"cellPoints.txt");

    // Open a file to write cell points
    OFstream cellPointsFile(cellPointsFilePath);

    // Get the boundary face information to exclude "Side" faces
    DynamicList<label> sideFaceLabels;
    forAll(boundaryMesh, patchI)
    {
        const polyPatch& patch = boundaryMesh[patchI];
        if (patch.name() == "Side")
        {
            forAll(patch, faceI)
            {
                sideFaceLabels.append(patch.start() + faceI);
            }
        }
    }

    const cellList& cells = mesh.cells();
    forAll(cells, cellI)
    {
        const cell& c = cells[cellI];
        forAll(c, faceI)
        {
            label faceLabel = c[faceI];
            if (!sideFaceLabels.contains(faceLabel))
            {
                cellFacesFile << faceLabel << " ";
            }
        }
        cellFacesFile << "\n";

        // Write the points of each cell, only saving unique points with z = 0
        std::unordered_set<label> uniquePoints;
        forAll(c, faceI)
        {
            const face& f = mesh.faces()[c[faceI]];
            forAll(f, pointI)
            {
                label pointLabel = f[pointI];
                const point& p = mesh.points()[pointLabel];
                if (p.z() == 0 && uniquePoints.find(pointLabel) == uniquePoints.end())
                {
                    uniquePoints.insert(pointLabel);
                    cellPointsFile << pointLabel << " ";
                }
            }
        }
        cellPointsFile << "\n";
    }

    // Get the mesh points
    const pointField& points = mesh.points();

    // Define the output file paths
    fileName pointIndexFilePath(runTime.path()/"constant"/"polyMesh"/"pointIndices.txt");
    fileName pointCoordinateFilePath(runTime.path()/"constant"/"polyMesh"/"pointCoordinates.txt");

    // Open files to write point data
    OFstream pointIndexFile(pointIndexFilePath);
    OFstream pointCoordinateFile(pointCoordinateFilePath);

    for (label pointI = 0; pointI < points.size(); ++pointI)
    {
        const point& p = points[pointI];

        if (p.z() == 0)
        {
            // Write point index
            pointIndexFile << pointI << "\n";

            // Write point coordinates
            pointCoordinateFile << p.x() << " " << p.y() << " " << p.z() << "\n";
        }
    }

    // Get the mesh points and faces
    const faceList& faces = mesh.faces();
    const surfaceVectorField& faceAreas = mesh.Sf(); // face area vectors
    const labelList& owner = mesh.owner();
    const labelList& neighbour = mesh.neighbour();

    // Define the output file paths
    fileName faceIndexFilePath(runTime.path()/"constant"/"polyMesh"/"faceIndices.txt");
    fileName facePointsFilePath(runTime.path()/"constant"/"polyMesh"/"facePoints.txt");
    fileName faceAreaFilePath(runTime.path()/"constant"/"polyMesh"/"faceAreas.txt");
    fileName faceVectorFilePath(runTime.path()/"constant"/"polyMesh"/"faceVectors.txt");
    fileName faceCellsFilePath(runTime.path()/"constant"/"polyMesh"/"faceCells.txt");

    // Open files to write face data
    OFstream faceIndexFile(faceIndexFilePath);
    OFstream facePointsFile(facePointsFilePath);
    OFstream faceAreaFile(faceAreaFilePath);
    OFstream faceVectorFile(faceVectorFilePath);
    OFstream faceCellsFile(faceCellsFilePath);

    for (label faceI = 0; faceI < faces.size(); ++faceI)
    {
        const face& f = faces[faceI];

        if (f.size() == 4)
        {
            // Find the two points with z = 0
            List<label> zeroZPoints;
            for (label i = 0; i < f.size(); ++i)
            {
                if (points[f[i]].z() == 0)
                {
                    zeroZPoints.append(f[i]);
                }
            }

            if (zeroZPoints.size() == 2)
            {
                const point& start = points[zeroZPoints[0]];
                const point& end = points[zeroZPoints[1]];

                vector faceVector = end - start;

                // Calculate the surface area of the face
                scalar area = mag(faceAreas[faceI]);

                // Write face index
                faceIndexFile << faceI << "\n";

                // Write the two points with z = 0
                facePointsFile << zeroZPoints[0] << " " << zeroZPoints[1] << "\n";

                // Write the surface area of the face
                faceAreaFile << area << "\n";

                // Write the vector between these two points
                faceVectorFile << faceVector.x() << " " << faceVector.y() << " " << faceVector.z() << "\n";

                // Write the cells connected to this face
                label ownerCell = owner[faceI];
                label neighbourCell = (neighbour.size() > faceI) ? neighbour[faceI] : -1;

                faceCellsFile << ownerCell << " " << neighbourCell << "\n";
            }
        }
    }

    // Iterate over the boundary patches to get face centers, cells, and face indices
    forAll(boundaryMesh, patchI)
    {
        const polyPatch& patch = boundaryMesh[patchI];

        // Skip patches named "Side"
        if (patch.name() == "Side") 
        {
            continue;
        }

        fileName faceCenterFilePath(runTime.path()/"constant"/"polyMesh"/("faceCenters_" + patch.name() + ".txt"));
        OFstream faceCenterFile(faceCenterFilePath);

        fileName faceCellFilePath(runTime.path()/"constant"/"polyMesh"/("faceCells_" + patch.name() + ".txt"));
        OFstream faceCellFile(faceCellFilePath);

        fileName boundaryFaceIndicesFilePath(runTime.path()/"constant"/"polyMesh"/("boundaryFaceIndices_" + patch.name() + ".txt"));
        OFstream boundaryFaceIndicesFile(boundaryFaceIndicesFilePath);

        forAll(patch, faceI)
        {
            const face& f = patch[faceI];
            vector faceCenter = f.centre(mesh.points());
            faceCenterFile << faceCenter.x() << " " << faceCenter.y() << " " << faceCenter.z() << "\n";

            label cellI = patch.faceCells()[faceI];
            faceCellFile << cellI << "\n";

            // Write the boundary face index
            boundaryFaceIndicesFile << (patch.start() + faceI) << "\n";
        }
    }

    // Define the output file path for boundary point indices
    fileName boundaryPointIndicesFilePath(runTime.path()/"constant"/"polyMesh"/"boundaryPointIndices.txt");

    // Open a file to write boundary point indices
    OFstream boundaryPointIndicesFile(boundaryPointIndicesFilePath);

    // Define a set to store unique boundary point indices
    std::unordered_set<label> boundaryPointLabels;

    forAll(boundaryMesh, patchI)
    {
        const polyPatch& patch = boundaryMesh[patchI];

        // Skip patches named "Side"
        if (patch.name() == "Side")
        {
            continue;
        }

        forAll(patch, faceI)
        {
            const face& f = patch[faceI];
            forAll(f, pointI)
            {
                label pointLabel = f[pointI];
                const point& p = mesh.points()[pointLabel];
                if (p.z() == 0)
                {
                    boundaryPointLabels.insert(pointLabel);
                }
            }
        }
    }

    // Write the boundary point indices to the file
    for (const auto& pointLabel : boundaryPointLabels)
    {
        boundaryPointIndicesFile << pointLabel << "\n";
    }

    // Define the output file path for boundary face indices
    fileName boundaryFaceIndicesFilePath(runTime.path()/"constant"/"polyMesh"/"boundaryFaces.txt");

    // Open a file to write boundary face indices
    OFstream boundaryFaceIndicesFile(boundaryFaceIndicesFilePath);

    // Define the output file path for boundary cell indices
    fileName boundaryCellIndicesFilePath(runTime.path()/"constant"/"polyMesh"/"boundaryCells.txt");

    // Open a file to write boundary cell indices
    OFstream boundaryCellIndicesFile(boundaryCellIndicesFilePath);

    forAll(boundaryMesh, patchI)
    {
        const polyPatch& patch = boundaryMesh[patchI];

        // Skip patches named "Side"
        if (patch.name() == "Side")
        {
            continue;
        }

        forAll(patch, faceI)
        {
            label faceLabel = patch.start() + faceI;
            boundaryFaceIndicesFile << faceLabel << "\n";

            // Get the cell to which this face is connected (owner cell)
            label ownerCell = owner[faceLabel];
            boundaryCellIndicesFile << ownerCell << "\n";
        }
    }

    // Define the output file path for boundary names excluding "Side"
fileName boundaryNamesFilePath(runTime.path()/"constant"/"polyMesh"/"boundaryNames.txt");

// Open a file to write boundary names excluding "Side"
OFstream boundaryNamesFile(boundaryNamesFilePath);

forAll(boundaryMesh, patchI)
{
    const polyPatch& patch = boundaryMesh[patchI];

    // Skip patches named "Side"
    if (patch.name() == "Side")
    {
        continue;
    }

    // Write the patch name to the file
    boundaryNamesFile << patch.name() << "\n";
}


    return 0;
}
