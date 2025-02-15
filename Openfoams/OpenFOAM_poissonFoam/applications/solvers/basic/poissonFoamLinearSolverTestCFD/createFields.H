/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Description
        Create and read the fields for the steady-state Poisson equation solver.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Info<< "Reading field p_zero_jacobi\n" << endl;
volScalarField p_zero_jacobi
(
    IOobject
    (
        "p_zero_jacobi",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field p_zero_GAMG\n" << endl;
volScalarField p_zero_GAMG
(
    IOobject
    (
        "p_zero_GAMG",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field p_model_jacobi\n" << endl;
volScalarField p_model_jacobi
(
    IOobject
    (
        "p_model_jacobi",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field p_model_GAMG\n" << endl;
volScalarField p_model_GAMG
(
    IOobject
    (
        "p_model_GAMG",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);
// #include "createPhi.H"

Info<< "Reading diffusivity tau\n" << endl;
volScalarField tau
(
    IOobject
    (
        "tau",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field source \n" << endl;
volScalarField source
(
    IOobject
    (
        "source",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

IOField<scalar> sourceField
(
    IOobject
    (
        "pEqnSource",               // Field name
        runTime.timeName(),         // Current time directory
        mesh,                       // Associated mesh
        IOobject::MUST_READ,        // Ensure it is read from file
        IOobject::AUTO_WRITE        // Automatically write after solving
    )
);

IOField<scalar> faceField
(
    IOobject
    (
        "pEqnFace",               // Field name
        runTime.timeName(),         // Current time directory
        mesh,                       // Associated mesh
        IOobject::MUST_READ,        // Ensure it is read from file
        IOobject::AUTO_WRITE        // Automatically write after solving
    )
);

IOField<scalar> dField
(
    IOobject
    (
        "pEqnD",               // Field name
        runTime.timeName(),         // Current time directory
        mesh,                       // Associated mesh
        IOobject::MUST_READ,        // Ensure it is read from file
        IOobject::AUTO_WRITE        // Automatically write after solving
    )
);

#include "createFvOptions.H"

// ************************************************************************* //
