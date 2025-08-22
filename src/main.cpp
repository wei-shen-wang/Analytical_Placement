#define _GLIBCXX_USE_CXX11_ABI 0
#include "Placement.h"
#include "Util.h"
#include "GlobalPlacer.h"
#include "arghandler.h"
#include "DPlace.h"
#include "TetrisLegal.h"
#include "ParamPlacement.h"

#include <iostream>
#include <string.h>
#include <time.h>

using namespace std;
bool handleArgument( const int& argc, char* argv[], CParamPlacement& param )
{

    int i;
    if( argc > 2 && strcmp( argv[1]+1, "aux" ) == 0 ){
        param.auxFilename = string( argv[2] );
        i = 3;
    }
    else{
        cout << "Usage: " << argv[0] << " -aux benchmark.aux" << endl;
        return false;
    }
    while( i < argc )
    {
        if( strlen( argv[i] ) <= 1 )
        {
            i++;
            continue;
        }

        ////////////////////////////////////////////////////////////////////////
        // Public arguments
        ////////////////////////////////////////////////////////////////////////
        else if( strcmp( argv[i]+1, "noglobal" ) == 0 )
            param.bRunGlobal = false;
        else if( strcmp( argv[i]+1, "nolegal" ) == 0 )
            param.bRunLegal = false;
        else if( strcmp( argv[i]+1, "nodetail" ) == 0 )
            param.bRunDetail = false;
        else if( strcmp( argv[i]+1, "loadpl" ) == 0 ){
            param.plFilename = string( argv[++i] );
        }
        i++;
    }
    return true;
}
///////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    
	
	gArg.Init( argc, argv );
    if( !handleArgument( argc, argv, param ) )
        return -1;



    Placement placement;
    placement.readBookshelfFormat(param.auxFilename , param.plFilename);
    double original_center_x = placement.rectangleChip().centerX();
    double original_center_y = placement.rectangleChip().centerY();
    placement.moveDesignCenter(-original_center_x, -original_center_y); // Move the design center to (0, 0)
    cout << "Benchmark: " << placement.name() << endl;

    cout << format("HPWL: %.f", placement.computeHpwl()) << endl;
    cout << format("Memory usage: %.1f MB", getCurrentMemoryUsage()) << endl;
    cout << format("Core region: (%.f,%.f)-(%.f,%.f)",
                   placement.boundryLeft(),
                   placement.boundryBottom(),
                   placement.boundryRight(),
                   placement.boundryTop()) << endl;
    double orig_wirelength;
    double gp_wirelength, lg_wirelength, dp_wirelength;
    bool bLegal=false;
    orig_wirelength=gp_wirelength=lg_wirelength=dp_wirelength=0;

    ////////////////////////////////////////////////////////////////
    // Global Placement
    ////////////////////////////////////////////////////////////////

    time_t total_time=0;
    time_t global_time_start = time(NULL);
    time_t total_global_time = 0;
    if(param.bRunGlobal)
    {

		
        cout<<endl<<"////// Global Placement ///////"<<endl;

		
        ////////////start to edit your code /////////////
		
		GlobalPlacer globalPlacer(placement);
		globalPlacer.place();
		globalPlacer.plotPlacementResult( placement.name() + "_gp.plt" );

		/////////////////////////////////////////////////
		
		
        placement.outputBookshelfFormat(placement.name()+".gp.pl");

        gp_wirelength = placement.computeHpwl();
        printf( "\nHPWL: %.0f\n",gp_wirelength);
        total_global_time = time(NULL) - global_time_start;
        total_time+=total_global_time;
    }
    placement.moveDesignCenter(original_center_x, original_center_y); // Move the design back


    ////////////////////////////////////////////////////////////////
    // Legalization
    ////////////////////////////////////////////////////////////////


    time_t legal_time_start = time(NULL);
    time_t total_legal_time = 0;
    if(param.bRunLegal){

        cout<<endl<<"////// Legalization ///////"<<endl;
        orig_wirelength = placement.computeHpwl();

        CTetrisLegal legal(placement);

        bLegal = legal.Solve( 0.8 );
        if(bLegal)
            cout<<"legalization success!"<<endl;
        else
            cout<<"legalization fail!"<<endl;


        placement.outputBookshelfFormat(placement.name()+".lg.pl");

        lg_wirelength = placement.computeHpwl();
        printf( "\nHPWL: %.0f (%3.2f%%)\n",
                lg_wirelength, ((lg_wirelength - orig_wirelength)/orig_wirelength)*100.0);
        total_legal_time = time(NULL) - legal_time_start;
        total_time+=total_legal_time;
    }

    ////////////////////////////////////////////////////////////////
    // Detail Placement
    ////////////////////////////////////////////////////////////////

    time_t detail_time_start = time(NULL);
    time_t total_detail_time = 0;
    if(param.bRunDetail && bLegal){

        cout<<endl<<"////// Detail Placement ///////"<<endl;
        orig_wirelength = placement.computeHpwl();

        CDetailPlacer dplacer( placement);
        dplacer.DetailPlace();


        placement.outputBookshelfFormat(placement.name()+".dp.pl");

        dp_wirelength = placement.computeHpwl();
        printf( "\nHPWL: %.0f (%3.2f%%)\n",
                dp_wirelength, ((dp_wirelength - orig_wirelength)/orig_wirelength)*100.0);
        total_detail_time = time(NULL) - detail_time_start;
        total_time+=total_detail_time;
    }
    GlobalPlacer plotter(placement);
    plotter.plotPlacementResult( placement.name() + "_final.plt" );

    cout<<endl<<endl<<"////////////////////"<<endl;
    if(placement.plname()!="")
        cout << "Benchmark: " << placement.plname() << endl;
    else
        cout << "Benchmark: " << placement.name() << endl;
    if(param.bRunGlobal)
        printf( "\nGlobal HPWL: %.0f   Time: %6.1f sec (%.1f min)\n",gp_wirelength, (double)total_global_time, (double)total_global_time/ 60.0);
    if(param.bRunLegal)
        printf( " Legal HPWL: %.0f   Time: %6.1f sec (%.1f min)\n",lg_wirelength, (double)total_legal_time, (double)total_legal_time/ 60.0);
    if(param.bRunDetail && bLegal)
        printf( "Detail HPWL: %.0f   Time: %6.1f sec (%.1f min)\n",dp_wirelength, (double)total_detail_time, (double)total_detail_time/ 60.0);
    printf( " ===================================================================\n" );
    printf( "       HPWL: %.0f   Time: %6.1f sec (%.1f min)\n", placement.computeHpwl(), (double)total_time, (double)total_time / 60.0 );

    // output the printed results to a file with same precison as above
    FILE *fp = fopen( (placement.name() + "_result.txt").c_str(), "w" );
    fprintf( fp, "Benchmark: %s\n", placement.name().c_str() );
    if(param.bRunGlobal)
        fprintf( fp, "Global HPWL: %.0f   Time: %6.1f sec (%.1f min)\n", gp_wirelength, (double)total_global_time, (double)total_global_time / 60.0);
    if(param.bRunLegal)
        fprintf( fp, " Legal HPWL: %.0f   Time: %6.1f sec (%.1f min)\n", lg_wirelength, (double)total_legal_time, (double)total_legal_time / 60.0);
    if(param.bRunDetail && bLegal)
        fprintf( fp, "Detail HPWL: %.0f   Time: %6.1f sec (%.1f min)\n", dp_wirelength, (double)total_detail_time, (double)total_detail_time / 60.0);
    fprintf( fp, " ===================================================================\n" );
    fprintf( fp, "       HPWL: %.0f   Time: %6.1f sec (%.1f min)\n", placement.computeHpwl(), (double)total_time, (double)total_time / 60.0 );
    fflush(fp);
    fclose(fp);
    return 0;
}
