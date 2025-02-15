volScalarField rAU(1.0/UEqn.A());
volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA));

if (pimple.ddtCorr())
{
    phiHbyA += MRF.zeroFilter(fvc::interpolate(rAU)*fvc::ddtCorr(U, phi, Uf));
}


MRF.makeRelative(phiHbyA);

if (p.needReference())
{
    fvc::makeRelative(phiHbyA, U);
    adjustPhi(phiHbyA, U, p);
    fvc::makeAbsolute(phiHbyA, U);
}

tmp<volScalarField> rAtU(rAU);

if (pimple.consistent())
{
    rAtU = 1.0/max(1.0/rAU - UEqn.H1(), 0.1/rAU);
    phiHbyA +=
        fvc::interpolate(rAtU() - rAU)*fvc::snGrad(p)*mesh.magSf();
    HbyA -= (rAU - rAtU())*fvc::grad(p);
}

if (pimple.nCorrPISO() <= 1)
{
    tUEqn.clear();
}

// Update the pressure BCs to ensure flux consistency
constrainPressure(p, U, phiHbyA, rAtU(), MRF);

// Non-orthogonal pressure corrector loop
// i = 0;
bool firstIteration = true; // Move this outside the loop

while (pimple.correctNonOrthogonal())
{
    volScalarField pInitial
    (
        IOobject
        (
            "pInitial",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        p
    );

    fvScalarMatrix pEqn
    (
        fvm::laplacian(rAtU(), p) == fvc::div(phiHbyA)
    );

    pEqn.setReference(pRefCell, pRefValue);
    auto start = std::chrono::high_resolution_clock::now();
    pEqn.solve(p.select(pimple.finalInnerIter()));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    Info << "Time taken to solve pEqn: " << elapsed.count() << " seconds" << endl;

    
    if (firstIteration && runTime.writeTime())
    {
        Info << "WRITE DATA TO FILE" << endl;


        volScalarField source_rec = fvc::laplacian(rAtU(), p);

        fvScalarMatrix pEqn2
        (
            fvm::laplacian(rAtU(), p) == source_rec
        );

        pEqn2.solve();

        volScalarField source_fvc
        (
            IOobject
            (
                "source_fvc",                      // Desired name
                runTime.timeName(),            // Current time directory
                mesh,                          // Mesh
                IOobject::NO_READ,             // Do not read from file
                IOobject::AUTO_WRITE           // Automatically write the file
            ),
            source_rec      // Field definition
        );

        source_fvc.write();

        volScalarField pAfter
        (
            IOobject
            (
                "pAfter",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::AUTO_WRITE
            ),
            p
        );


        rAtU().write();
        volScalarField tau
        (
            IOobject
            (
                "tau",                      // Desired name
                runTime.timeName(),            // Current time directory
                mesh,                          // Mesh
                IOobject::NO_READ,             // Do not read from file
                IOobject::AUTO_WRITE           // Automatically write the file
            ),
            rAtU()      // Field definition
        );
        tau.write();

        pInitial.write();
        pAfter.write();


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
            pEqn2.D()
        );
        DField.write();

        IOField<scalar> UpperField
        (
            IOobject
            (
                "pEqnFace",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            pEqn2.upper()
        );
        UpperField.write();

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
            pEqn2.source()
        );
        sourceField.write();

        firstIteration = false; // Set the flag to false after the first iteration
    }

    if (pimple.finalNonOrthogonalIter())
    {
        phi = phiHbyA - pEqn.flux();
    }
}



#include "continuityErrs.H"

// Explicitly relax pressure for momentum corrector
p.relax();

U = HbyA - rAtU*fvc::grad(p);
U.correctBoundaryConditions();
fvOptions.correct(U);

// Correct Uf if the mesh is moving
fvc::correctUf(Uf, U, phi);

// Make the fluxes relative to the mesh motion
fvc::makeRelative(phi, U);
