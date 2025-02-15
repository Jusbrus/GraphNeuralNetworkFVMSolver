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

    volScalarField source = fvc::laplacian(tau, p_input);

    tau.write();

    fvScalarMatrix pEqn
    (
        fvm::laplacian(tau, p) == source
    );

        // Info << pEqn << " seconds" << endl;

    Info << "pEqn.source()" << pEqn.source() << endl;

    Info << "pEqn.upper()" << pEqn.upper() << endl;

    Info << "pEqn.D()" << pEqn.D() << endl;



    fvOptions.constrain(pEqn);

    // Timing the pEqn.solve() call
    auto start = std::chrono::high_resolution_clock::now();
    pEqn.solve();

    Info << "p" << p<< endl;

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    Info << "Time taken to solve pEqn: " << elapsed.count() << " seconds" << endl;

    volScalarField source_rec = fvc::laplacian(tau, p);

    fvScalarMatrix pEqn2
    (
        fvm::laplacian(tau, p) == source_rec
    );

    Info << "pEqn2.source()" << pEqn2.source() << endl;

    Info << "pEqn2.upper()" << pEqn2.upper() << endl;

    Info << "pEqn2.D()" << pEqn2.D() << endl;

    Info << "p2" << p<< endl;





    // Timing the pEqn2.solve() call
    start = std::chrono::high_resolution_clock::now();
    pEqn2.solve();
    end = std::chrono::high_resolution_clock::now();

    elapsed = end - start;
    Info << "Time taken to solve pEqn2: " << elapsed.count() << " seconds" << endl;

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

    Info << "Current simulation time: " << runTime.timeName() << endl;
    #include "write.H"

    Info<< "End\n" << endl;

    return 0;
}

// ************************************************************************* //
