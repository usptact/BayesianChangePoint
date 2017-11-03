using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;

namespace BayesianChangePoint
{
    class Program
    {
        static void Main(string[] args)
        {
            /* Original data with missing values
            int[] disaster_data = new int[] {4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                                             3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                                             2, 2, 3, 4, 2, 1, 3, -999, 2, 1, 1, 1, 1, 3, 0, 0,
                                             1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                                             0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                                             3, 3, 1, -999, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                                             0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1 };
            */                                 

            int[] disaster_data = new int[] {4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                                             3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                                             2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                                             1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                                             0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                                             3, 3, 1, 2, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                                             0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1 };

            int[] years_data = Enumerable.Range(0, 111).ToArray();

            Range n = new Range(disaster_data.Length).Named("years_idx");

            var switchpoint = Variable.DiscreteUniform(n).Named("switchpoint");

            var early_rate = Variable.Exp(1).Named("early_rate");
            var late_rate = Variable.Exp(1).Named("late_rate"); ;

            var data = Variable.Array<int>(n).Named("data");
            var years = Variable.Array<int>(n).Named("years");

            using (Variable.ForEach(n))
            {
                using (Variable.If(switchpoint > years[n]))
                    data[n] = Variable.Poisson(late_rate);
                using (Variable.IfNot(switchpoint > years[n]))
                    data[n] = Variable.Poisson(early_rate);
            }

            data.ObservedValue = disaster_data;
            years.ObservedValue = years_data;

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.GenerateInMemory = false;
            engine.Compiler.WriteSourceFiles = true;
            engine.Compiler.IncludeDebugInformation = true;

            try
            {
                var switchpointMarginal = engine.Infer<Discrete>(switchpoint);
                Console.WriteLine(switchpointMarginal);
            }
            catch (MicrosoftResearch.Infer.Maths.AllZeroException exception)
            {
                Console.WriteLine(exception);
            }

            Console.ReadKey();
        }
    }
}
