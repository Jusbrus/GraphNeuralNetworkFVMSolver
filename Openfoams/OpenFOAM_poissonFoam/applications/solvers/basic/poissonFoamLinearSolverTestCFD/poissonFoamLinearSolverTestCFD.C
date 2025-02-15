/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    poissonFoam

Group
    grpBasicSolvers

Description
    Steady-state Poisson equation solver for a scalar quantity.

    \heading Solver details
    The solver is applicable to, e.g. electrostatic potential or heat distribution.
    The equation is given by:

    \f[
        \div \left( \tau \grad p \right) = \text{source}
    \f]

    Where:
    \vartable
        p     | Scalar field which is solved for
        \tau  | Diffusion coefficient
        \text{source} | Source term
    \endvartable

    \heading Required fields
    \plaintable
        p     | Scalar field which is solved for
    \endplaintable

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "fvOptions.H"
#include "simpleControl.H"
#include "OFstream.H"
#include <chrono>  // Include this header for high-resolution clock

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Steady-state Poisson equation solver for a scalar quantity."
    );

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"

    simpleControl simple(mesh);

    #include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nCalculating scalar field p\n" << endl;

    
    runTime++;

    // volScalarField source_rec = fvc::laplacian(tau, p_model_jacobi);
    

    fvScalarMatrix pEqn_model_jacobi
    (
        fvm::laplacian(tau, p_model_jacobi) == source
    );

    pEqn_model_jacobi.source() = sourceField;
    pEqn_model_jacobi.upper() = faceField;
    pEqn_model_jacobi.D() = dField;

    fvScalarMatrix pEqn_model_GAMG
    (
        fvm::laplacian(tau, p_model_GAMG) == source
    );

    pEqn_model_GAMG.source() = sourceField;
    pEqn_model_GAMG.upper() = faceField;
    pEqn_model_GAMG.D() = dField;

    fvScalarMatrix pEqn_zero_jacobi
    (
        fvm::laplacian(tau, p_zero_jacobi) == source
    );

    pEqn_zero_jacobi.source() = sourceField;
    pEqn_zero_jacobi.upper() = faceField;
    pEqn_zero_jacobi.D() = dField;

    fvScalarMatrix pEqn_zero_GAMG
    (
        fvm::laplacian(tau, p_zero_GAMG) == source
    );

    pEqn_zero_GAMG.source() = sourceField;
    pEqn_zero_GAMG.upper() = faceField;
    pEqn_zero_GAMG.D() = dField;


    fvOptions.constrain(pEqn_zero_jacobi);
    fvOptions.constrain(pEqn_model_jacobi);
    fvOptions.constrain(pEqn_zero_GAMG);
    fvOptions.constrain(pEqn_model_GAMG);

    // Timing each solve call
    auto start = std::chrono::high_resolution_clock::now();
    pEqn_model_jacobi.solve();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    Info << "Time taken to solve pEqn_model_jacobi: " << elapsed.count() << " seconds" << endl;

    start = std::chrono::high_resolution_clock::now();
    pEqn_zero_jacobi.solve();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    Info << "Time taken to solve pEqn_zero_jacobi: " << elapsed.count() << " seconds" << endl;

    start = std::chrono::high_resolution_clock::now();
    pEqn_model_GAMG.solve();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    Info << "Time taken to solve pEqn_model_GAMG: " << elapsed.count() << " seconds" << endl;


    start = std::chrono::high_resolution_clock::now();
    pEqn_zero_GAMG.solve();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    Info << "Time taken to solve pEqn_zero_GAMG: " << elapsed.count() << " seconds" << endl;

    // Info<< p_model_jacobi << endl;

    Info << "Current simulation time: " << runTime.timeName() << endl;
    #include "write.H"

    Info<< "End\n" << endl;

    return 0;
}

// ************************************************************************* //
