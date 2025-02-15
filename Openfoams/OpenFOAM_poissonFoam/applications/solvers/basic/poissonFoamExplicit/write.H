/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Description
        Write the pressure field.

\*---------------------------------------------------------------------------*/

// p.write();

if (runTime.writeTime())
{
    p.write();

    // Write the contents of pEqn fields
    // IOField<scalar> pField
    // (
    //     IOobject
    //     (
    //         "p_python",
    //         runTime.timeName(),
    //         mesh,
    //         IOobject::NO_READ,
    //         IOobject::AUTO_WRITE
    //     ),
    //     p.internalField()
    // );
    // pField.write();

    // Write the contents of pEqn fields
    IOField<scalar> DField
    (
        IOobject
        (
            "pEqnD",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        pEqn.D()
    );
    DField.write();

    IOField<scalar> upperField

    (
        IOobject
        (
            "pEqnFace",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        pEqn.upper()
    );
    upperField.write();

    IOField<scalar> sourceField
    (
        IOobject
        (
            "pEqnSource",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        pEqn.source()
    );
    sourceField.write();
}

// ************************************************************************* //
